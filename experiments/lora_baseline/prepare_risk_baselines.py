"""Verify published fixed-C10 artifacts and export private paired-comparison CSVs.

This does not construct risk training inputs, refit any model or issue a gate.
The publication evidence is the saved build/fixed_c10_analysis.json, which binds
the final manifest and patient-score table to the manuscript revision.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_record_v1 import sha256, verify

DISEASES = dict(ckd='chronic_kidney_disease', fatty_liver='fatty_liver', diabetes='diabetes')


def prepare(fixed, evidence_path, out):
    fixed, evidence_path, out = map(lambda p: Path(p).resolve(), (fixed, evidence_path, out))
    if any((p / '.git').exists() for p in (out, *out.parents)):
        raise ValueError('Patient-level export must be outside a Git checkout')
    staging = out.with_name(out.name + '.preparing')
    if out.exists() or staging.exists():
        raise ValueError('Output or staging directory already exists')
    import pandas as pd
    evidence = json.loads(evidence_path.read_text())
    manifest_hash = verify(fixed / 'final_manifest.json', evidence['manifest_sha256'])
    scores_path = fixed / 'restricted_patient_level/fixed_patient_scores.parquet'
    score_hash = verify(scores_path, evidence['patient_scores_sha256'])
    manifest = json.loads((fixed / 'final_manifest.json').read_text())
    if manifest['status'] != 'COMPLETE' or manifest['fixed_C'] != 10:
        raise ValueError('Incomplete or non-C10 final analysis')
    metric_name = 'aggregate/five_year_model_metrics.csv'
    verify(fixed / metric_name, manifest['outputs'][metric_name])
    metrics = pd.read_csv(fixed / metric_name)
    scores = pd.read_parquet(scores_path)
    if len(scores) != manifest['patients'] or not scores.person_id.is_unique:
        raise ValueError('Published patient score cohort mismatch')
    models = {}
    for disease, phenotype in DISEASES.items():
        for kind in ('age_sex_utilization', 'fermat_patient_representation'):
            path = fixed / 'models' / f'{phenotype}__{kind}.json'
            model = json.loads(path.read_text())
            if model.get('fixed_C') != 10 or model['test_future_outcomes_used']:
                raise ValueError('Model is not a fixed-C10 training-only artifact')
            models[f'{disease}/{kind}'] = dict(sha256=sha256(path), fixed_C=model['fixed_C'],
                train_eligible=model['train_eligible_n'], train_observed=model['train_observed_n'],
                validation_eligible=model['validation_eligible_n'], validation_observed=model['validation_observed_n'])
        p = scores[f'{phenotype}__fermat_patient_representation_5y'].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid published probabilities')
    staging.mkdir(parents=True)
    for disease, phenotype in DISEASES.items():
        pd.DataFrame(dict(patient_id=scores.person_id,
                         probability=scores[f'{phenotype}__fermat_patient_representation_5y'])).to_csv(
                             staging / f'{disease}_v1_probabilities.csv', index=False)
    report = dict(status='BASELINES_PREPARED_NOT_REPRODUCED', patients=len(scores),
                  publication_evidence_sha256=sha256(evidence_path), manifest_sha256=manifest_hash,
                  patient_scores_sha256=score_hash, models=models,
                  published_metrics=metrics.loc[metrics.model == 'fermat_patient_representation',
                      ['disease', 'eligible_n', 'observed_status_n', 'ipcw_roc_auc', 'ipcw_brier']].to_dict('records'),
                  risk_training_dataset_ready=False)
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    staging.rename(out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed-dir', type=Path, required=True)
    p.add_argument('--publication-evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.fixed_dir, args.publication_evidence, args.out), indent=2))


if __name__ == '__main__':
    main()
