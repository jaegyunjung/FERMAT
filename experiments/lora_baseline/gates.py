"""Versioned reproduction evidence. No training or CLI side effects on import."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL_VERSION = '2026-10-22.5'
PROTOCOL_TARGETS = Path(__file__).with_name('protocol_targets.json')
GATE_METRICS = dict(record=('test.recall_at_10',), risk=('test.weighted_auroc', 'test.weighted_brier'))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def official_targets(task, disease):
    table = json.loads(PROTOCOL_TARGETS.read_text())
    if table['protocol_version'] != PROTOCOL_VERSION:
        raise ValueError('Protocol code and target file versions differ')
    targets = table[task] if task == 'record' else table[task][disease]
    return normalize_targets(targets)


def normalize_targets(targets):
    """Explanatory notes are not thresholds and must not affect their identity."""
    return {metric: {k: float(target[k]) for k in ('expected', 'tolerance')}
            for metric, target in targets.items()}


def metric_check(metric, value, expected, tolerance):
    value, expected, tolerance = map(float, (value, expected, tolerance))
    passed = (all(math.isfinite(v) for v in (value, expected, tolerance))
              and tolerance >= 0 and abs(value - expected) <= tolerance)
    return dict(metric=metric, value=value, expected=expected, tolerance=tolerance, passed=passed)


def published_risk_fit(fit):
    return fit == dict(selection='fixed', candidates=[10.0], steps_per_fit=200, selected_C=10.0)


def make_gate(log, targets):
    targets = normalize_targets(targets)
    summary, fingerprint = log['summary'], log['fingerprint']
    if summary['arm'] != 'head_only':
        raise ValueError('The gate is defined on a head_only reproduction run')
    identity = fingerprint['gate_identity']
    task, disease = identity['task'], identity['disease']
    checks = []
    for metric, target in targets.items():
        value = summary
        for key in metric.split('.'):
            value = value[key]
        checks.append(metric_check(metric, value, **target))
    missing = sorted(set(GATE_METRICS[task]) - targets.keys())
    custom = targets != official_targets(task, disease)
    trained = summary.get('optimizer_steps', 0) > 0 and summary['selected']['epoch'] > 0
    recipe_ok = fingerprint.get('recipe') == 'v1'
    risk_fit = None
    if task == 'risk':
        risk_fit = dict(fingerprint.get('risk_regularization') or {}, selected_C=summary['selected'].get('C'))
        recipe_ok = recipe_ok and published_risk_fit(risk_fit) and summary.get('optimizer_steps') == 200
    return dict(passed=trained and not missing and all(c['passed'] for c in checks)
                and (identity['smoke'] or (recipe_ok and not custom)),
                risk_regularization=risk_fit,
                custom_targets=custom, targets_sha256=content_hash(targets),
                recipe=fingerprint.get('recipe'), missing_required_metrics=missing,
                checks=checks, identity=identity, selected_epoch=summary['selected']['epoch'],
                optimizer_steps=summary.get('optimizer_steps'),
                summary_git_commit=fingerprint.get('git_commit'))


def validate_gate(gate, identity):
    reasons = []
    if not gate.get('passed'):
        reasons.append('gate did not pass')
    if gate.get('optimizer_steps', 0) <= 0 or gate.get('selected_epoch', 0) <= 0:
        reasons.append('gate has no completed training update')
    g = gate.get('identity', {})
    for key in ('protocol_version', 'task', 'disease', 'smoke', 'model_sha256',
                'dataset_metadata_sha256', 'dataset_content_sha256', 'precision',
                'validation_patient_ids_sha256', 'test_patient_ids_sha256', 'label_definition_sha256'):
        if identity.get(key) is None and key not in ('disease', 'model_sha256'):
            reasons.append(f'{key}: missing run identity')
        elif key not in g or g[key] != identity.get(key):
            reasons.append(f'{key}: gate and run differ')
    if identity['protocol_version'] != PROTOCOL_VERSION:
        reasons.append('protocol_version: obsolete protocol')
    checks = gate.get('checks', [])
    by_metric = {c.get('metric'): c for c in checks}
    if len(by_metric) != len(checks):
        reasons.append('duplicate gate metrics')
    targets = {}
    for metric, check in by_metric.items():
        try:
            checked = metric_check(metric, check['value'], check['expected'], check['tolerance'])
            targets[metric] = {k: checked[k] for k in ('expected', 'tolerance')}
            if not checked['passed']:
                reasons.append(f'{metric}: failed or nonfinite measurement')
        except (KeyError, TypeError, ValueError):
            reasons.append(f'{metric}: malformed measurement')
    for metric in GATE_METRICS[identity['task']]:
        if metric not in by_metric:
            reasons.append(f'required metric {metric} not checked by the gate')
    try:
        if gate.get('targets_sha256') != content_hash(targets):
            reasons.append('target content hash mismatch')
    except ValueError:
        reasons.append('nonfinite target values')
    if not identity['smoke']:
        if gate.get('custom_targets') or targets != official_targets(identity['task'], identity['disease']):
            reasons.append('real runs require the official protocol targets')
        if gate.get('recipe') != 'v1':
            reasons.append('real runs require a v1 reproduction recipe')
        if identity['task'] == 'risk' and (not published_risk_fit(gate.get('risk_regularization'))
                                          or gate.get('optimizer_steps') != 200):
            reasons.append('real risk gates require fixed C=10 with 200 steps, without grid selection')
    same_list = g.get('train_patient_ids_sha256') == identity.get('train_patient_ids_sha256')
    # Full dataset content is compared above, including the entire training pool.
    if not (same_list or g.get('training_selection') == 'all'):
        reasons.append('gate training patients are neither the same list nor the full pool')
    return reasons
