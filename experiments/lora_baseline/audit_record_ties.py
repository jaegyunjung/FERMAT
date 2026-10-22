"""Apply historical patient-seeded and stable tie rules to identical saved-head logits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import PrefixSplit
from prepare_record_v1 import sha256
from runtime_guard import RuntimeGuard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'head', 'standardization', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--gpu-profile', type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh audit directory')
    args.out.mkdir(parents=True)
    guard = RuntimeGuard(args.gpu_profile, args.out)
    split = PrefixSplit(args.data, 'test')
    classes = np.load(args.data / 'classes.npy')
    z = np.load(args.data / 'test/frozen_embeddings.npy')
    scale = np.load(args.standardization)
    head = torch.nn.Linear(z.shape[1], len(classes)).to(args.device)
    head.load_state_dict(torch.load(args.head, map_location='cpu', weights_only=True)['state'])
    head.eval()
    x = torch.from_numpy((z - scale['mean']) / scale['sd']).to(args.device)
    seeded, stable, changed, straddles = [], [], 0, 0
    with torch.inference_mode():
        for start in range(0, len(split), 256):
            guard.check()
            scores = head(x[start:start+256]).cpu().numpy()
            for j, score in enumerate(scores):
                i = start + j
                truth = set(map(int, split.label_lists([i])[0]))
                priority = np.random.default_rng(20260908 + int(split.patient_id[i])).random(len(score))
                historical = classes[np.lexsort((priority, -score))[:10]]
                deterministic = classes[np.argsort(-score, kind='stable')[:10]]
                seeded.append(100 * len(truth.intersection(historical)) / len(truth))
                stable.append(100 * len(truth.intersection(deterministic)) / len(truth))
                changed += set(historical) != set(deterministic)
                boundary = np.partition(score, -10)[-10]
                straddles += int((score > boundary).sum() < 10 < (score >= boundary).sum())
    report = dict(patients=len(split), device=args.device,
                  source_sha256=dict(head=sha256(args.head), standardization=sha256(args.standardization),
                                     embeddings=sha256(args.data / 'test/frozen_embeddings.npy')),
                  historical_recall_percent=float(np.mean(seeded)), stable_recall_percent=float(np.mean(stable)),
                  stable_minus_historical_pp=float(np.mean(np.array(stable) - seeded)),
                  changed_prediction_sets=int(changed), boundary_ties=int(straddles),
                  maximum_patient_recall_difference_pp=float(np.abs(np.array(stable) - seeded).max()))
    guard.check(force=True)
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
