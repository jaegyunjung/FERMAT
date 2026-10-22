"""Reproduction gate: compare a head_only run against the manuscript v1 numbers.

Writes gate.json with ``passed`` true or false and exits non-zero on failure.
The LoRA arm of each task script refuses to start without a passed gate, so
the protocol's order (reproduce first, then fine-tune) is enforced by code.

    python check_gate.py --summary runs/record/head_only_s1/summary.json \
        --metric test.recall_at_10 --expected 55.68 --tolerance 1.0 --out runs/record/gate.json
    python check_gate.py --summary runs/risk/ckd/head_only_full/summary.json \
        --metric test.weighted_auroc --expected 0.871 --tolerance 0.01 \
        --metric test.weighted_brier --expected 0.0204 --tolerance 0.002 --out runs/risk/ckd/gate.json
"""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--metric', action='append', required=True, help='Dotted key inside summary, e.g. test.recall_at_10')
parser.add_argument('--expected', action='append', type=float, required=True)
parser.add_argument('--tolerance', action='append', type=float, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if not len(args.metric) == len(args.expected) == len(args.tolerance):
    raise SystemExit('--metric, --expected and --tolerance must be repeated the same number of times')
log = json.loads(args.summary.read_text())
summary = log['summary']
if summary['arm'] != 'head_only':
    raise SystemExit('The gate is defined on a head_only reproduction run')
checks = []
for metric, expected, tolerance in zip(args.metric, args.expected, args.tolerance):
    value = summary
    for key in metric.split('.'):
        value = value[key]
    checks.append(dict(metric=metric, value=float(value), expected=expected, tolerance=tolerance,
                       passed=abs(float(value) - expected) <= tolerance))
result = dict(passed=all(c['passed'] for c in checks), checks=checks, summary=str(args.summary),
              summary_git_commit=log['fingerprint'].get('git_commit'),
              train_patient_ids_sha256=log['fingerprint'].get('train_patient_ids_sha256'))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
sys.exit(0 if result['passed'] else 1)
