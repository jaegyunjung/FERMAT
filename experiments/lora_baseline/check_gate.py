"""Reproduction gate: compare a head_only run against the protocol's v1 targets.

Writes gate.json carrying the run identity (task, disease, protocol version,
model hash, dataset and patient-list hashes, label-definition hash, training
selection) and every checked metric with its target, tolerance and value. The
LoRA arm of each task script refuses to start unless a gate passed *and* its
identity matches the run, so the protocol's order (reproduce first, then
fine-tune) is enforced by code.

Targets come from protocol_targets.json. ``--expected``/``--tolerance`` overrides
exist for synthetic pipeline tests only; a gate built with them is marked
``custom_targets`` and is rejected by any run on real weights.

    python check_gate.py --summary runs/record/head_only_s1/summary.json --out runs/record/gate.json
    python check_gate.py --summary runs/risk/ckd/head_only_full/summary.json --out runs/risk/ckd/gate.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--targets', type=Path, default=common.PROTOCOL_TARGETS)
parser.add_argument('--metric', action='append', default=None, help='Test-only override; repeatable')
parser.add_argument('--expected', action='append', type=float, default=None)
parser.add_argument('--tolerance', action='append', type=float, default=None)
args = parser.parse_args()

log = json.loads(args.summary.read_text())
summary, fingerprint = log['summary'], log['fingerprint']
if summary['arm'] != 'head_only':
    raise SystemExit('The gate is defined on a head_only reproduction run')
identity = fingerprint['gate_identity']
task, disease = identity['task'], identity['disease']

custom = args.metric is not None
if custom:
    if not (args.expected and args.tolerance and len(args.metric) == len(args.expected) == len(args.tolerance)):
        raise SystemExit('--metric, --expected and --tolerance must be repeated the same number of times')
    targets = {m: dict(expected=e, tolerance=t) for m, e, t in zip(args.metric, args.expected, args.tolerance)}
else:
    table = json.loads(args.targets.read_text())
    if table.get('protocol_version') != common.PROTOCOL_VERSION:
        raise SystemExit(f'{args.targets} is for protocol {table.get("protocol_version")}, code is {common.PROTOCOL_VERSION}')
    targets = table[task] if task == 'record' else table[task].get(disease)
    if targets is None:
        raise SystemExit(f'No protocol targets for task {task} disease {disease}')

checks = []
for metric, target in targets.items():
    value = summary
    for key in metric.split('.'):
        value = value[key]
    checks.append(dict(metric=metric, value=float(value), expected=float(target['expected']),
                       tolerance=float(target['tolerance']),
                       passed=abs(float(value) - float(target['expected'])) <= float(target['tolerance'])))
missing = [m for m in common.GATE_METRICS[task] if m not in targets]
result = dict(
    passed=all(c['passed'] for c in checks) and not missing,
    custom_targets=custom,
    missing_required_metrics=missing,
    checks=checks,
    identity=identity,
    summary=str(args.summary),
    summary_git_commit=fingerprint.get('git_commit'),
    selected_epoch=summary['selected']['epoch'],
    optimizer_steps=summary.get('optimizer_steps'),
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result['passed'] else 1)
