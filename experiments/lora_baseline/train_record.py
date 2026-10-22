"""Task R: diagnoses, prescriptions and procedures recorded on the first clinical
date after the observed history (manuscript Section 2.3), with the frozen
representation arm and the LoRA fine-tuning arm trained in one pipeline.

head_only: frozen last-event hidden state -> standardised -> linear multilabel
           head, trained on cached embeddings with the manuscript recipe.
lora:      LoRA adapters in the transformer and the same linear head trained
           jointly end to end on the same patients and targets.
Selection uses validation Recall@10 with validation BCE as tie-breaker; the
test split is scored once with the selected state.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from data import PrefixSplit, class_lookup, fixed_subset, iterate_batches, multi_hot, read_metadata  # noqa: E402
from metrics import record_metrics, paired_bootstrap, mean_difference, rank_classes  # noqa: E402

parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
parser.add_argument('--classes', type=Path, default=None,
                    help='int64 .npy of predicted token IDs; defaults to <data>/classes.npy, then the release head classes')
parser.add_argument('--baseline-recall', type=Path, default=None,
                    help='CSV with patient_id,recall_at_10 from a saved arm for the paired bootstrap')
parser.add_argument('--baseline-label', default='baseline')
args = parser.parse_args()
defaults = dict(head_only=dict(epochs=60, patience=10, lr_head=1e-3, weight_decay=0.01, batch=256),
                lora=dict(epochs=20, patience=5, lr_head=1e-3, weight_decay=0.01, batch=args.batch_size))[args.arm]
epochs = args.epochs or defaults['epochs']
patience = args.patience or defaults['patience']
lr_head = args.lr_head or defaults['lr_head']
weight_decay = defaults['weight_decay'] if args.weight_decay is None else args.weight_decay
resamples = args.bootstrap_resamples or 5000
common.set_seed(args.seed)
args.out.mkdir(parents=True, exist_ok=True)

model, model_info = common.build_model(args)
splits = {name: PrefixSplit(args.data, name, model.config.block_size) for name in ('train', 'validation', 'test')}
for name, split in splits.items():
    if split.labels is None:
        raise ValueError(f'{name}: record-task labels are missing')
classes_path = args.classes or (args.data / 'classes.npy' if (args.data / 'classes.npy').exists()
                                else args.release / 'artifacts/head_classes.npy')
classes = np.load(classes_path, allow_pickle=False).astype(np.int64)
class_index = class_lookup(classes, model.config.vocab_size)
n_classes = len(classes)
train_rows = fixed_subset(len(splits['train']), args.train_subset, args.subset_seed)
val_rows, test_rows = np.arange(len(splits['validation'])), np.arange(len(splits['test']))
log = dict(fingerprint=common.fingerprint(args, model_info, read_metadata(args.data), dict(
    classes=str(classes_path), n_classes=n_classes, n_train=len(train_rows),
    n_validation=len(val_rows), n_test=len(test_rows),
    train_patient_ids_sha256=__import__('hashlib').sha256(splits['train'].patient_id[train_rows].tobytes()).hexdigest(),
    epochs=epochs, patience=patience, lr_head=lr_head, weight_decay=weight_decay)), epochs=[])

# Standardisation always comes from the frozen model on the training patients,
# so both arms feed the head inputs on the same scale (MODEL_CARD.md).
frozen = {name: common.frozen_embeddings(model, splits[name], rows, args)
          for name, rows in (('train', train_rows), ('validation', val_rows), ('test', test_rows))}
mean, sd = common.standardization(frozen['train'])
mean_t, sd_t = (torch.from_numpy(a).to(args.device) for a in (mean, sd))
snapshot, wrapped = common.prepare_adapters(model, args)
head = torch.nn.Linear(model.config.n_embd, n_classes).to(args.device)
optimizer, set_warmup = common.make_optimizer(model, head, args, lr_head, weight_decay)
targets = {name: multi_hot(splits[name].label_lists(rows), class_index, n_classes)
           for name, rows in (('train', train_rows), ('validation', val_rows))}


def scores_for(name, rows, cached):
    """Head logits for a split; cached embeddings for head_only, live forward for lora."""
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_rows in iterate_batches(np.arange(len(rows)), args.batch_size):
            if cached is not None:
                z = torch.from_numpy(cached[batch_rows]).to(args.device)
            else:
                idx, age, typ = splits[name].batch(rows[batch_rows], args.device)
                with common.autocast(args):
                    z = common.last_hidden(model, idx, age, typ).float()
            outputs.append(head((z - mean_t) / sd_t).float().cpu())
    return torch.cat(outputs)


def validate():
    logits = scores_for('validation', val_rows, frozen['validation'] if args.arm == 'head_only' else None)
    bce = float(F.binary_cross_entropy_with_logits(logits, targets['validation'], reduction='none').sum(1).mean())
    truth = targets['validation'] > 0
    top = rank_classes(logits, 10)
    hits = torch.gather(truth, 1, top).sum(1).float()
    recall = float((100. * hits / truth.sum(1)).mean())
    return recall, bce


best, best_state, since_best, step = None, None, 0, 0
rng = np.random.default_rng(args.seed)
for epoch in range(1, epochs + 1):
    head.train()
    model.train() if args.arm == 'lora' else model.eval()
    order = rng.permutation(len(train_rows))
    losses = []
    batch_size = defaults['batch']
    for n_batch, batch_rows in enumerate(iterate_batches(order, batch_size)):
        y = targets['train'][batch_rows].to(args.device)
        if args.arm == 'head_only':
            z = torch.from_numpy(frozen['train'][batch_rows]).to(args.device)
        else:
            idx, age, typ = splits['train'].batch(train_rows[batch_rows], args.device)
            with common.autocast(args):
                z = common.last_hidden(model, idx, age, typ).float()
        loss = F.binary_cross_entropy_with_logits(head((z - mean_t) / sd_t), y)
        (loss / args.grad_accum).backward()
        losses.append(float(loss.detach()))
        if (n_batch + 1) % args.grad_accum == 0:
            set_warmup(step)
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                           args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
    optimizer.zero_grad(set_to_none=True)
    recall, bce = validate()
    log['epochs'].append(dict(epoch=epoch, train_bce=float(np.mean(losses)), validation_recall_at_10=recall,
                              validation_bce=bce, optimizer_steps=step))
    print(json.dumps(log['epochs'][-1]), flush=True)
    improved = best is None or recall > best[0] or (recall == best[0] and bce < best[1])
    if improved:
        best, best_state, since_best = (recall, bce, epoch), common.trainable_state(model, head), 0
    else:
        since_best += 1
        if since_best >= patience:
            break

common.restore_state(model, head, best_state)
test_logits = scores_for('test', test_rows, frozen['test'] if args.arm == 'head_only' else None)
per_patient = record_metrics(test_logits, splits['test'].label_lists(test_rows), class_index)
patient_ids = splits['test'].patient_id[test_rows]
with (args.out / 'test_patient_metrics.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['patient_id'] + list(per_patient))
    for row in range(len(patient_ids)):
        writer.writerow([int(patient_ids[row])] + [f'{per_patient[k][row]:.6f}' for k in per_patient])
torch.save(dict(patient_id=torch.from_numpy(patient_ids), top10_classes=classes[rank_classes(test_logits, 10).numpy()]),
           args.out / 'test_top10.pt')
summary = dict(arm=args.arm, selected=dict(epoch=best[2], validation_recall_at_10=best[0], validation_bce=best[1]),
               test={k: float(v.mean()) for k, v in per_patient.items()},
               lora=dict(wrapped_modules=wrapped, rank=args.rank, alpha=args.alpha, targets=args.targets) if wrapped else None,
               audit=common.finish_audit(model, snapshot, head, args, args.out))
if args.baseline_recall is not None:
    baseline = common.align_baseline(patient_ids, common.read_baseline_csv(args.baseline_recall, 'recall_at_10'),
                                     args.baseline_label)
    summary['paired_recall_at_10_minus_' + args.baseline_label] = paired_bootstrap(
        mean_difference, [per_patient['recall_at_10'], baseline], resamples, args.seed)
log['summary'] = summary
common.write_json(args.out / 'summary.json', log)
print(json.dumps(summary, indent=2, default=float))
