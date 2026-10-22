"""Task D: five-year disease risk for one disease (manuscript Section 2.2 detailed
comparison), with the frozen representation arm and the LoRA arm in one pipeline.

head_only: frozen hidden state and the four clinical variables -> weighted
           logistic predictor fitted with the manuscript recipe (AdamW, 200
           full-batch steps, learning rate 0.03, weight decay 1e-5).
lora:      LoRA adapters and the same logistic head trained jointly with
           mini-batches; the epoch with the lowest validation weighted BCE is
           selected and the test split is scored once.
Weights are the supplied clipped inverse observation probabilities; patients
with unobserved outcomes carry weight zero and never enter a loss or metric.
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
from data import PrefixSplit, fixed_subset, iterate_batches, read_metadata  # noqa: E402
from metrics import weighted_auroc, weighted_brier, paired_bootstrap  # noqa: E402

parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
parser.add_argument('--disease', required=True, help='Suffix of outcome_<disease>.npy / weight_<disease>.npy')
parser.add_argument('--baseline-probabilities', type=Path, default=None,
                    help='CSV with patient_id,probability from a saved arm for the paired bootstrap')
parser.add_argument('--baseline-label', default='baseline')
args = parser.parse_args()
defaults = dict(head_only=dict(epochs=1, patience=1, lr_head=0.03, weight_decay=1e-5, steps=200),
                lora=dict(epochs=3, patience=1, lr_head=1e-3, weight_decay=0.01))[args.arm]
epochs = args.epochs or defaults['epochs']
patience = args.patience or defaults['patience']
lr_head = args.lr_head or defaults['lr_head']
weight_decay = defaults['weight_decay'] if args.weight_decay is None else args.weight_decay
resamples = args.bootstrap_resamples or 2000
common.set_seed(args.seed)
args.out.mkdir(parents=True, exist_ok=True)

model, model_info = common.build_model(args)
splits = {name: PrefixSplit(args.data, name, model.config.block_size, diseases=[args.disease])
          for name in ('train', 'validation', 'test')}
for name, split in splits.items():
    if split.clinical is None:
        raise ValueError(f'{name}: clinical variables are missing')
observed = {name: np.flatnonzero(split.outcomes[args.disease][1] > 0) for name, split in splits.items()}
train_rows = observed['train'][fixed_subset(len(observed['train']), args.train_subset, args.subset_seed)]
val_rows, test_rows = observed['validation'], np.arange(len(splits['test']))
if len(train_rows) == 0 or len(val_rows) == 0:
    raise ValueError('No observed outcomes to train or validate on')


def outcome(name, rows):
    y, w = splits[name].outcomes[args.disease]
    return (torch.from_numpy(y[rows].astype(np.float32)).clamp_min(0.),
            torch.from_numpy(w[rows].astype(np.float32)))


# Clinical variables: training-set median imputation, then mean/sd scaling (Appendix A).
raw_clinical = np.asarray(splits['train'].clinical[train_rows], dtype=np.float64)
medians = np.nanmedian(raw_clinical, 0)


def clinical(name, rows):
    values = np.asarray(splits[name].clinical[rows], dtype=np.float64)
    values = np.where(np.isnan(values), medians, values)
    return values.astype(np.float32)


clin_mean, clin_sd = common.standardization(clinical('train', train_rows))
log = dict(fingerprint=common.fingerprint(args, model_info, read_metadata(args.data), dict(
    disease=args.disease, n_train_observed=len(train_rows), n_validation_observed=len(val_rows),
    n_test=len(test_rows), n_test_observed=int(len(observed['test'])),
    train_patient_ids_sha256=__import__('hashlib').sha256(splits['train'].patient_id[train_rows].tobytes()).hexdigest(),
    epochs=epochs, patience=patience, lr_head=lr_head, weight_decay=weight_decay)), epochs=[])

frozen = {name: common.frozen_embeddings(model, splits[name], rows, args)
          for name, rows in (('train', train_rows), ('validation', val_rows), ('test', test_rows))}
emb_mean, emb_sd = common.standardization(frozen['train'])
scale = {k: torch.from_numpy(v).to(args.device) for k, v in
         dict(emb_mean=emb_mean, emb_sd=emb_sd, clin_mean=clin_mean, clin_sd=clin_sd).items()}
snapshot, wrapped = common.prepare_adapters(model, args)
head = torch.nn.Linear(model.config.n_embd + 4, 1).to(args.device)
optimizer, set_warmup = common.make_optimizer(model, head, args, lr_head, weight_decay)
clinical_t = {name: torch.from_numpy(clinical(name, rows)) for name, rows in
              (('train', train_rows), ('validation', val_rows), ('test', test_rows))}


def features(z, c):
    return torch.cat([(z - scale['emb_mean']) / scale['emb_sd'], (c - scale['clin_mean']) / scale['clin_sd']], 1)


def weighted_bce(logits, y, w):
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction='none')).sum() / w.sum()


def probabilities(name, rows, cached):
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch_rows in iterate_batches(np.arange(len(rows)), args.batch_size):
            c = clinical_t[name][batch_rows].to(args.device)
            if cached is not None:
                z = torch.from_numpy(cached[batch_rows]).to(args.device)
            else:
                idx, age, typ = splits[name].batch(rows[batch_rows], args.device)
                with common.autocast(args):
                    z = common.last_hidden(model, idx, age, typ).float()
            outputs.append(torch.sigmoid(head(features(z, c)).squeeze(1)).float().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def validate():
    p = probabilities('validation', val_rows, frozen['validation'] if args.arm == 'head_only' else None)
    y, w = outcome('validation', val_rows)
    y, w = y.numpy(), w.numpy()
    bce = float((w * -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))).sum() / w.sum())
    return bce, weighted_auroc(y, p, w), weighted_brier(y, p, w)


y_train, w_train = outcome('train', train_rows)
best, best_state, since_best, step = None, None, 0, 0
rng = np.random.default_rng(args.seed)
if args.arm == 'head_only':
    # Full-batch logistic fit on cached embeddings: the manuscript's recipe, no selection.
    z = torch.from_numpy(frozen['train']).to(args.device)
    x = features(z, clinical_t['train'].to(args.device))
    y, w = y_train.to(args.device), w_train.to(args.device)
    for step in range(defaults['steps']):
        loss = weighted_bce(head(x).squeeze(1), y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    bce, auroc, brier = validate()
    log['epochs'].append(dict(epoch=1, train_weighted_bce=float(loss.detach()), validation_weighted_bce=bce,
                              validation_weighted_auroc=auroc, validation_weighted_brier=brier, optimizer_steps=defaults['steps']))
    best, best_state = (bce, 1), common.trainable_state(model, head)
else:
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        losses = []
        for n_batch, batch_rows in enumerate(iterate_batches(rng.permutation(len(train_rows)), args.batch_size)):
            idx, age, typ = splits['train'].batch(train_rows[batch_rows], args.device)
            with common.autocast(args):
                z = common.last_hidden(model, idx, age, typ).float()
            logits = head(features(z, clinical_t['train'][batch_rows].to(args.device))).squeeze(1)
            loss = weighted_bce(logits, y_train[batch_rows].to(args.device), w_train[batch_rows].to(args.device))
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
        bce, auroc, brier = validate()
        log['epochs'].append(dict(epoch=epoch, train_weighted_bce=float(np.mean(losses)), validation_weighted_bce=bce,
                                  validation_weighted_auroc=auroc, validation_weighted_brier=brier, optimizer_steps=step))
        print(json.dumps(log['epochs'][-1]), flush=True)
        if best is None or bce < best[0]:
            best, best_state, since_best = (bce, epoch), common.trainable_state(model, head), 0
        else:
            since_best += 1
            if since_best >= patience:
                break

common.restore_state(model, head, best_state)
p_test = probabilities('test', test_rows, frozen['test'] if args.arm == 'head_only' else None)
y_all, w_all = splits['test'].outcomes[args.disease]
patient_ids = splits['test'].patient_id[test_rows]
with (args.out / 'test_patient_probabilities.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['patient_id', 'probability', 'outcome', 'weight'])
    for row in range(len(patient_ids)):
        writer.writerow([int(patient_ids[row]), f'{p_test[row]:.8f}', int(y_all[row]), f'{w_all[row]:.6f}'])
mask = w_all > 0
y_obs, w_obs, p_obs = y_all[mask].astype(np.float64), w_all[mask].astype(np.float64), p_test[mask]
summary = dict(arm=args.arm, disease=args.disease, selected=dict(epoch=best[1], validation_weighted_bce=best[0]),
               test=dict(weighted_auroc=weighted_auroc(y_obs, p_obs, w_obs), weighted_brier=weighted_brier(y_obs, p_obs, w_obs),
                         observed=int(mask.sum()), cases=int((y_obs == 1).sum()), mean_probability=float(p_obs.mean())),
               lora=dict(wrapped_modules=wrapped, rank=args.rank, alpha=args.alpha, targets=args.targets) if wrapped else None,
               audit=common.finish_audit(model, snapshot, head, args, args.out))
if args.baseline_probabilities is not None:
    baseline = common.align_baseline(patient_ids, common.read_baseline_csv(args.baseline_probabilities, 'probability'),
                                     args.baseline_label)[mask]
    summary['paired_minus_' + args.baseline_label] = dict(
        auroc=paired_bootstrap(lambda y, a, b, w: weighted_auroc(y, a, w) - weighted_auroc(y, b, w),
                               [y_obs, p_obs, baseline, w_obs], resamples, args.seed),
        brier=paired_bootstrap(lambda y, a, b, w: weighted_brier(y, a, w) - weighted_brier(y, b, w),
                               [y_obs, p_obs, baseline, w_obs], resamples, args.seed))
log['summary'] = summary
common.write_json(args.out / 'summary.json', log)
print(json.dumps(summary, indent=2, default=float))
