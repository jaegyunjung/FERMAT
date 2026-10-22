"""Issue reproduction evidence from a head_only summary, never from a LoRA run."""
import argparse
import json
from pathlib import Path
from gates import PROTOCOL_VERSION, PROTOCOL_TARGETS, make_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--targets', type=Path, default=PROTOCOL_TARGETS)
    parser.add_argument('--metric', action='append')
    parser.add_argument('--expected', action='append', type=float)
    parser.add_argument('--tolerance', action='append', type=float)
    args = parser.parse_args()
    log = json.loads(args.summary.read_text())
    identity = log['fingerprint']['gate_identity']
    if identity['protocol_version'] != PROTOCOL_VERSION:
        parser.error('Summary uses an obsolete protocol; rerun the reproduction')
    if args.metric is not None:
        if not (args.expected and args.tolerance and len(args.metric) == len(args.expected) == len(args.tolerance)):
            parser.error('Repeat --metric, --expected and --tolerance together')
        if len(set(args.metric)) != len(args.metric):
            parser.error('Duplicate metric overrides')
        targets = {m: dict(expected=e, tolerance=t) for m, e, t in zip(args.metric, args.expected, args.tolerance)}
    else:
        if args.expected is not None or args.tolerance is not None:
            parser.error('--expected/--tolerance require --metric')
        table = json.loads(args.targets.read_text())
        if table['protocol_version'] != PROTOCOL_VERSION:
            parser.error('Target file uses an obsolete protocol')
        targets = table['record'] if identity['task'] == 'record' else table['risk'][identity['disease']]
    result = make_gate(log, targets)
    result['summary'] = str(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
