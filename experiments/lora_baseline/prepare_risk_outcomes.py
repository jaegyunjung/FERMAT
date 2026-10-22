"""Audit saved C10 predictors and stage private risk outcomes, without GPU/DB use.

No prefix export, fresh embeddings, model refit or reproduction gate is made.
Historical observation weights were not saved as arrays in the supplied
manifest. Reconstruct them with the hash-verified original function and record
that provenance explicitly; never label them bitwise-verified saved weights.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn

from prepare_record_v1 import sha256, verify
from prepare_risk_baselines import DISEASES
from recipes import fit_risk_scaler, risk_features, risk_probabilities


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(root, evidence_path, core_path, out):
    root, evidence_path, core_path, out = map(Path, (root, evidence_path, core_path, out))
    out = out.resolve()
    require(not any((p / '.git').exists() for p in (out, *out.parents)), 'Private output must be outside Git')
    staging = out.with_name(out.name + '.preparing')
    require(not out.exists() and not staging.exists(), 'Output/staging already exists')
    start = time.perf_counter()
    fixed = root / 'fixed_c10_20260909'
    manifest = json.loads((root / 'comparators/manifest.json').read_text())
    evidence = json.loads(evidence_path.read_text())
    fixed_manifest = json.loads((fixed / 'final_manifest.json').read_text())
    hashes = {'core': verify(core_path, manifest['core_script_sha256']),
              'fixed_manifest': verify(fixed / 'final_manifest.json', evidence['manifest_sha256']),
              'scores': verify(fixed / 'restricted_patient_level/fixed_patient_scores.parquet', evidence['patient_scores_sha256'])}
    require(fixed_manifest['status'] == 'COMPLETE' and fixed_manifest['fixed_C'] == 10, 'Not a final C10 analysis')
    for name, item in manifest['inputs'].items():
        hashes[name] = verify(item['path'], item['sha256'])
    spec = importlib.util.spec_from_file_location('original_risk_core', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    cohort = pd.read_parquet(manifest['inputs']['preindex']['path'])
    outcomes = pd.read_parquet(manifest['inputs']['outcomes']['path'])
    require(not outcomes['split'].eq('test').any(), 'Test outcomes in training source')
    test_path = root / 'final_evaluation_v2_raw_dx/restricted_patient_level/selected_test_outcomes_raw_dx_once.parquet'
    marker = json.loads((test_path.parent / 'raw_dx_outcome_open_marker.json').read_text())
    hashes['test_outcomes'] = verify(test_path, marker['sealed_sha256'])
    outcomes = pd.concat([outcomes, pd.read_parquet(test_path)], ignore_index=True)
    frame = cohort.merge(outcomes, on=['person_id', 'split'], validate='one_to_one')
    require(len(frame) == len(cohort) and frame.person_id.is_unique, 'Cohort join lost or duplicated patients')
    frame['log1p_clinical_rows_through_index'] = np.log1p(frame.clinical_rows_through_index)
    frame['log1p_clinical_active_days_through_index'] = np.log1p(frame.clinical_active_days_through_index)
    frames = {name: frame.loc[frame.split.eq(old)].reset_index(drop=True)
              for name, old in [('train', 'train'), ('validation', 'val'), ('test', 'test')]}
    # Read only held-out embeddings here. Full fresh extraction is a separate audit.
    emb = pd.read_parquet(manifest['inputs']['embeddings']['path'], filters=[('split', '==', 'test')])
    embcols = sorted(c for c in emb if c.startswith('emb_'))
    test = frames['test'].merge(emb[['person_id', *embcols]], on='person_id', validate='one_to_one')
    scores = pd.read_parquet(fixed / 'restricted_patient_level/fixed_patient_scores.parquet').set_index('person_id')
    require(set(test.person_id) == set(scores.index) and len(test) == 10000, 'Test population changed')
    metric_path = fixed / 'aggregate/five_year_model_metrics.csv'
    hashes['metrics'] = verify(metric_path, fixed_manifest['outputs']['aggregate/five_year_model_metrics.csv'])
    expected_metrics = pd.read_csv(metric_path)
    staging.mkdir(parents=True)
    for name, data in frames.items():
        folder = staging / name
        folder.mkdir()
        np.save(folder / 'patient_id.npy', data.person_id.to_numpy(np.int64))
        np.save(folder / 'clinical.npy', data[list(core.BASE_FEATURES)].to_numpy(np.float64))
    report = dict(status='OUTCOMES_PREPARED_NOT_TRAINING_READY', source_sha256=hashes,
                  preparation_script_sha256=sha256(__file__),
                  environment=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, sklearn=sklearn.__version__),
                  weight_provenance='reconstructed_original_ipcw_function; no saved weight arrays to compare',
                  split_counts={k: len(v) for k, v in frames.items()}, diseases={},
                  risk_training_dataset_ready=False, cuda_parity_verified=False, gate_issued=False)
    for disease, phenotype in DISEASES.items():
        path = fixed / 'models' / f'{phenotype}__fermat_patient_representation.json'
        artifact = json.loads(path.read_text())
        require(artifact['fixed_C'] == 10 and not artifact['test_future_outcomes_used'], 'Invalid model recipe')
        counts, statuses, weights = {}, {}, {}
        for name, data in frames.items():
            status = core.observed_status(data, phenotype, 5)
            weight = core.ipcw(data, status, phenotype)['weights']
            eligible = status.eligible.to_numpy(bool)
            observed = (status.eligible & status.observed).to_numpy(bool)
            y = np.where(observed, status.disease_event.to_numpy(np.int8), -1).astype(np.int8)
            require(np.array_equal(weight > 0, observed), 'Observation/weight mismatch')
            for stem, array in [('eligible', eligible), ('outcome', y), ('weight', weight)]:
                np.save(staging / name / f'{stem}_{disease}.npy', array)
            counts[name] = dict(eligible=int(eligible.sum()), observed=int(observed.sum()), cases=int((y == 1).sum()))
            statuses[name], weights[name] = status, weight
        for name, prefix in [('train', 'train'), ('validation', 'validation')]:
            for field in ('eligible', 'observed'):
                require(counts[name][field] == artifact[f'{prefix}_{field}_n'], f'{disease}: cohort count mismatch')
        pre = artifact['transformer']
        require(pre['columns'] == [*core.BASE_FEATURES, *embcols], 'Feature order mismatch')
        clinical_stats = fit_risk_scaler(frames['train'].loc[statuses['train'].eligible, list(core.BASE_FEATURES)].to_numpy(np.float64))
        errors = {key: float(np.max(np.abs(value - np.asarray(pre[old])[:4])))
                  for key, old, value in zip(('median', 'mean', 'sd'), ('medians', 'means', 'scales'), clinical_stats)}
        require(all(v == 0 for v in errors.values()), f'{disease}: clinical preprocessing mismatch: {errors}')
        scale = {f'{prefix}_{key}': torch.tensor(np.asarray(pre[old])[sl], dtype=torch.float64)
                 for prefix, sl in [('clinical', slice(0, 4)), ('embedding', slice(4, None))]
                 for key, old in [('median', 'medians'), ('mean', 'means'), ('sd', 'scales')]}
        z = torch.from_numpy(test[embcols].to_numpy(np.float64, copy=True))
        c = torch.from_numpy(test[list(core.BASE_FEATURES)].to_numpy(np.float64, copy=True))
        feature_error = float(np.max(np.abs(risk_features(z, c, scale).numpy() - core.transform(test, pre).astype(np.float32))))
        require(feature_error == 0, 'FP32 feature parity failed')
        head = torch.nn.Linear(len(pre['columns']), 1)
        with torch.no_grad():
            head.weight.copy_(torch.tensor(artifact['coefficients']).reshape(1, -1))
            head.bias.fill_(artifact['intercept'])
            p = risk_probabilities(head, z, c, scale).numpy()
        reference = scores.loc[test.person_id, f'{phenotype}__fermat_patient_representation_5y'].to_numpy(float)
        np.save(staging / 'test' / f'baseline_probability_{disease}.npy', reference)
        probability_error = float(np.max(np.abs(p - reference)))
        require(probability_error <= 1e-6, 'Saved probability replay failed')
        # Patient order remains that of frames['test']; merge above preserves it.
        require(np.array_equal(test.person_id, frames['test'].person_id), 'Test order changed')
        status, weight = statuses['test'], weights['test']
        y = status.disease_event.to_numpy(int)
        metrics = dict(auroc=core.weighted_roc_auc(y, reference, weight), brier=core.weighted_brier(y, reference, weight))
        row = expected_metrics.loc[(expected_metrics.disease == phenotype) & (expected_metrics.model == 'fermat_patient_representation')].iloc[0]
        metric_errors = dict(auroc=abs(metrics['auroc'] - row.ipcw_roc_auc), brier=abs(metrics['brier'] - row.ipcw_brier))
        require(max(metric_errors.values()) <= 1e-8, f'{disease}: reconstructed weight metrics differ: {metric_errors}')
        report['diseases'][disease] = dict(counts=counts, model_sha256=sha256(path), clinical_stat_max_errors=errors,
            fp32_features_max_error=feature_error, probability_max_error=probability_error,
            published_metrics_replayed=metrics, metric_max_errors=metric_errors)
        print(json.dumps(dict(disease=disease, **report['diseases'][disease])), flush=True)
    report['elapsed_seconds'] = time.perf_counter() - start
    report['array_sha256'] = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*.npy'))}
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    staging.rename(out)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'publication-evidence', 'original-core', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.publication_evidence, args.original_core, args.out)
