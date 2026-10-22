"""Reproduce a trained arm's test predictions from trainable_state.pt alone.

Loads the frozen release (or a smoke model), re-injects the adapters and head,
applies the saved standardisation, and writes the same per-patient outputs as
the training script. Used by the tests to prove a run is reproducible from its
saved files in a separate process.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from recipes import risk_probabilities
from data import PrefixSplit, class_lookup, iterate_batches  # noqa: E402
from metrics import record_metrics, rank_classes  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True, help='trainable_state.pt from a training run')
    parser.add_argument('--release', type=Path, default=common.DEFAULT_RELEASE)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bf16', action='store_true', default=None, help='Precision defaults to the checkpoint')
    parser.add_argument('--smoke-random-init', type=Path, default=None)
    parser.add_argument('--smoke-seed', type=int, default=1, help='Seed used to build the smoke model in training')
    parser.add_argument('--reference-scores', type=Path, help='Compare record validation outputs with the saved live-model scores')
    parser.add_argument('--gpu-profile', type=Path, help='Enable current-boot WSL GPU/host guard')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.out}; choose a new run directory')
    args.out.mkdir(parents=True, exist_ok=True)
    from runtime_guard import RuntimeGuard
    guard = RuntimeGuard(args.gpu_profile, args.out)
    if args.smoke_random_init is not None:
        common.set_seed(args.smoke_seed)
    checkpoint = torch.load(args.state, map_location='cpu', weights_only=True)
    saved = checkpoint['config']
    if saved.get('validation_only') and args.split != 'validation':
        raise ValueError('Validation-only checkpoint cannot score the test split')
    if args.reference_scores is not None and (args.split != 'validation' or saved['task'] != 'record'):
        raise ValueError('Reference-score audit is restricted to record validation')
    if args.reference_scores is not None and args.batch_size != saved.get('inference_batch_size', args.batch_size):
        raise ValueError('Reference-score audit requires the original inference batch size')
    if saved.get('protocol_version') != common.PROTOCOL_VERSION:
        raise ValueError('Checkpoint protocol differs; use the matching code version')
    if args.bf16 is not None and args.bf16 != saved['bf16']:
        raise ValueError('Requested precision differs from the checkpoint')
    args.bf16 = saved['bf16']
    if args.bf16 and not str(args.device).startswith('cuda'):
        raise ValueError('This checkpoint requires CUDA BF16 inference')
    args.smoke_seed = saved['smoke_seed']
    model, info = common.build_model(args)
    if info['model_sha256'] != saved['model_sha256'] or info['smoke'] != saved['smoke']:
        raise ValueError('Base model does not match the trained checkpoint')
    state, head = common.load_trained(args.state, model, args)
    config, pre = state['config'], state['preprocessing']
    task = config['task']
    split = PrefixSplit(args.data, args.split, model.config.block_size, diseases=[config['disease']] if task == 'risk' else ())
    rows = np.arange(len(split))
    emb_mean, emb_sd = (pre['embedding_mean'].to(args.device), pre['embedding_sd'].to(args.device))
    outputs = []
    with torch.no_grad():
        for batch_rows in iterate_batches(rows, args.batch_size):
            guard.check()
            idx, age, typ = split.batch(batch_rows, args.device)
            with common.autocast(args):
                z = common.last_hidden(model, idx, age, typ).float()
            x = (z - emb_mean) / emb_sd
            if task == 'risk':
                c = torch.from_numpy(np.asarray(split.clinical[batch_rows], dtype=np.float64)).to(args.device)
                outputs.append(risk_probabilities(head, z, c, pre).cpu())
            else:
                outputs.append(head(x).float().cpu())
    scores = torch.cat(outputs)
    patient_ids = split.patient_id[rows]
    if task == 'record':
        classes = pre['classes'].numpy()
        class_index = class_lookup(classes, model.config.vocab_size)
        output = dict(patient_id=torch.from_numpy(patient_ids.copy()), logits=scores,
                      top10_classes=torch.from_numpy(classes[rank_classes(scores, 10).numpy()]))
        if args.reference_scores is not None:
            reference = torch.load(args.reference_scores, map_location='cpu', weights_only=True)
            matching = output['logits'].shape == reference['logits'].shape
            report = dict(patient_order_exact=torch.equal(output['patient_id'], reference['patient_id']),
                          top10_exact=torch.equal(output['top10_classes'], reference['top10_classes']),
                          logits_allclose=matching and torch.allclose(scores, reference['logits'], rtol=1e-5, atol=1e-5),
                          logits_exact=torch.equal(scores, reference['logits']),
                          maximum_absolute_error=float((scores - reference['logits']).abs().max()) if matching else None,
                          rtol=1e-5, atol=1e-5)
            report['passed'] = report['patient_order_exact'] and report['top10_exact'] and report['logits_allclose']
            common.write_json(args.out / 'reload_audit.json', report)
            if not report['passed']:
                raise ValueError('Independent validation checkpoint reload differs from the live model')
        torch.save(output, args.out / f'{args.split}_scores.pt')
        if split.labels is not None:
            per_patient = record_metrics(scores, split.label_lists(rows), class_index)
            with (args.out / f'{args.split}_patient_metrics.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['patient_id'] + list(per_patient))
                for row in range(len(patient_ids)):
                    writer.writerow([int(patient_ids[row])] + [f'{per_patient[k][row]:.6f}' for k in per_patient])
            print(json.dumps({k: float(v.mean()) for k, v in per_patient.items()}))
    else:
        p = scores.numpy().astype(np.float64)
        y, w = split.outcomes[config['disease']]
        with (args.out / 'test_patient_probabilities.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['patient_id', 'probability', 'outcome', 'weight'])
            for row in range(len(patient_ids)):
                writer.writerow([int(patient_ids[row]), f'{p[row]:.8f}', int(y[row]), f'{w[row]:.6f}'])
        print(json.dumps(dict(n=len(p), mean_probability=float(p.mean()))))
    guard.check(force=True)


if __name__ == '__main__':
    main()
