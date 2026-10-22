"""Compare risk recipe primitives with local v1 source using synthetic data only.

python audit_v1.py --reference-dir /path/to/original/scripts --out /tmp/audit.json
No original main(), database queries, patient files or GPU jobs are executed.
Only the named pure preprocessing / fit / predict functions are extracted.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from recipes import fit_risk_scaler, risk_features, risk_probabilities, fit_risk_candidate, select_risk_candidate, RISK_C_VALUES
from metrics import weighted_brier


def reference_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'{path}: expected functions {names}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(reference_dir):
    import pandas as pd  # optional audit-only dependency, present in the v1 environment
    namespace = dict(np=np, torch=torch, pd=pd)
    hashes = {}
    for file, names in (
        ('task30_d3_analysis_core.py', ('fit_transformer', 'transform')),
        ('train_task30_d3_comparators.py', ('fit_weighted_torch', 'predict')),
    ):
        hashes[file] = reference_functions(reference_dir / file, names, namespace)
    rng = np.random.default_rng(42)
    clinical = rng.normal(size=(40, 4))
    clinical[1, 2] = np.nan
    clinical[2, 3] = np.inf
    embedding = rng.normal(size=(40, 6)).astype(np.float32)
    embedding[:, 0] *= 1e-5
    train, observed, val = np.arange(24), np.arange(18), np.arange(24, 40)
    frame = pd.DataFrame(np.column_stack([clinical, embedding]), columns=[f'x{i}' for i in range(10)])
    transformer = namespace['fit_transformer'](frame.iloc[train], list(frame.columns))
    reference_x = namespace['transform'](frame, transformer)
    pre = {}
    for prefix, values in (('clinical', clinical[train]), ('embedding', embedding[train])):
        for key, value in zip(('median', 'mean', 'sd'), fit_risk_scaler(values)):
            pre[f'{prefix}_{key}'] = torch.from_numpy(value)
    z, c = torch.from_numpy(embedding), torch.from_numpy(clinical)
    x = risk_features(z, c, pre)
    np.testing.assert_array_equal(x.numpy(), reference_x.astype(np.float32))
    y = (np.arange(40) % 3 == 0).astype(float)
    weights = rng.uniform(.5, 3., size=40)
    prevalence = float(np.average(y[observed], weights=weights[observed]))
    candidates, reference_candidates = [], []
    max_parameter_error, max_probability_error = 0., 0.
    for c_value in RISK_C_VALUES:
        coefficients, intercept = namespace['fit_weighted_torch'](
            reference_x[observed], y[observed], weights[observed], c_value, 'cpu', 200, .03)
        head = torch.nn.Linear(x.shape[1], 1)
        fit_risk_candidate(head, x[observed], torch.tensor(y[observed], dtype=torch.float32),
                           torch.tensor(weights[observed], dtype=torch.float32), prevalence, c_value)
        actual_coefficients = head.weight.detach().numpy().ravel()
        np.testing.assert_array_equal(actual_coefficients, coefficients)
        assert head.bias.item() == intercept
        with torch.no_grad():
            p = risk_probabilities(head, z[val], c[val], pre).numpy()
        reference_p = namespace['predict'](reference_x[val], coefficients, intercept)
        np.testing.assert_allclose(p, reference_p, rtol=0, atol=1e-12)
        max_parameter_error = max(max_parameter_error, float(np.max(np.abs(actual_coefficients - coefficients))))
        max_probability_error = max(max_probability_error, float(np.max(np.abs(p - reference_p))))
        candidates.append(dict(C=c_value, validation_weighted_brier=weighted_brier(y[val], p, weights[val])))
        reference_candidates.append(dict(C=c_value, validation_weighted_brier=weighted_brier(y[val], reference_p, weights[val])))
    chosen = select_risk_candidate(candidates)
    assert chosen == select_risk_candidate(reference_candidates)
    return dict(synthetic_only=True, actual_v1_reproduction=False, source_sha256=hashes,
                torch=str(torch.__version__), preprocessing_fp32_exact=True,
                max_parameter_error=max_parameter_error, max_probability_error=max_probability_error,
                selected_C=candidates[chosen]['C'], candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.reference_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
