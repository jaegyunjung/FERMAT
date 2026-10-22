"""Task D: five-year disease risk for one disease (manuscript Section 2.2 detailed
comparison), with the frozen representation arm and the LoRA arm in one pipeline.

head_only: frozen hidden state and the four clinical variables -> weighted
           logistic predictor fitted with the manuscript recipe (AdamW, 200
           full-batch steps per C, learning rate 0.03, validation Brier selection).
lora:      LoRA adapters and the same logistic head trained jointly with
           mini-batches; the epoch with the lowest validation weighted BCE is
           selected and the test split is scored once. Requires a passed
           reproduction gate (--gate) from a head_only run.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from data import PrefixSplit, iterate_batches, read_metadata  # noqa: E402
from metrics import weighted_auroc, weighted_brier, paired_bootstrap  # noqa: E402

from recipes import (configure_recipe, fit_risk_scaler, risk_features, risk_probabilities,
                     select_risk_candidate, fit_risk_candidate, RISK_C_VALUES)

def main():
    parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--disease', required=True, help='Suffix of outcome_<disease>.npy / weight_<disease>.npy')
    parser.add_argument('--baseline-probabilities', type=Path, default=None,
                        help='CSV with patient_id,probability from a saved arm for the paired bootstrap')
    parser.add_argument('--baseline-label', default='baseline')
    parser.add_argument('--risk-c-values', type=float, nargs='+', default=list(RISK_C_VALUES))
    parser.add_argument('--risk-steps', type=int, default=200)
    args = parser.parse_args()
    if args.arm == 'head_only' and args.weight_decay is not None:
        parser.error('Risk head_only derives weight decay from --risk-c-values; do not pass --weight-decay')
    if args.arm == 'head_only' and (args.grad_clip is not None or args.warmup_steps not in (None, 0)):
        parser.error('Full-batch risk head_only does not use clipping or warm-up')
    cfg = common.resolve_defaults(args, dict(
        head_only=dict(epochs=1, patience=1, lr_head=0.03, weight_decay=1e-5, steps=args.risk_steps),
        lora=dict(epochs=3, patience=1, lr_head=1e-3, weight_decay=0.01)))
    resamples = args.bootstrap_resamples or 2000
    configure_recipe(args, cfg, 'risk')
    if args.risk_steps <= 0 or any(not np.isfinite(c) or c <= 0 for c in args.risk_c_values):
        raise ValueError('Risk steps and C values must be positive and finite')
    if args.recipe == 'v1' and args.smoke_random_init is None:
        if args.risk_steps != 200 or tuple(args.risk_c_values) != RISK_C_VALUES:
            raise ValueError('v1 requires 200 steps and the original C grid')
    common.set_seed(args.seed)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.out}; choose a new run directory')
    args.out.mkdir(parents=True, exist_ok=True)

    model, model_info = common.build_model(args)
    splits = {name: PrefixSplit(args.data, name, model.config.block_size, diseases=[args.disease])
              for name in ('train', 'validation', 'test')}
    common.check_split_overlap(splits)
    for name, split in splits.items():
        if split.clinical is None:
            raise ValueError(f'{name}: clinical variables are missing')
    observed = {name: np.flatnonzero(split.outcomes[args.disease][1] > 0) for name, split in splits.items()}


    class ObservedPool:
        """Restrict training selection to patients with an observed outcome."""

        def __init__(self, split, rows):
            self.patient_id = split.patient_id[rows]
            self.rows = rows

        def __len__(self):
            return len(self.rows)


    pool = ObservedPool(splits['train'], observed['train'])
    chosen, selection_mode = common.select_training_rows(pool, args)
    train_rows = pool.rows[chosen]
    val_rows, test_rows = observed['validation'], np.arange(len(splits['test']))
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError('No observed outcomes to train or validate on')
    if args.arm == 'head_only' and args.grad_accum != 1:
        raise ValueError('--grad-accum does not apply to the full-batch head_only fit')
    label_hash = common.sha256_of_arrays(*splits['validation'].outcomes[args.disease], *splits['test'].outcomes[args.disease])
    identity = common.gate_identity('risk', args.disease, model_info, splits, label_hash, train_rows, selection_mode,
                                    read_metadata(args.data), precision='bf16' if args.bf16 else 'float32')
    gate = common.require_gate(args, identity)


    def outcome(name, rows):
        y, w = splits[name].outcomes[args.disease]
        return (torch.from_numpy(y[rows].astype(np.float32)).clamp_min(0.),
                torch.from_numpy(w[rows].astype(np.float32)))


    # v1 preprocessing includes every eligible training patient, including censored patients.
    pre_rows = np.flatnonzero(splits['train'].eligible[args.disease])

    def clinical(name, rows):
        return np.asarray(splits[name].clinical[rows], dtype=np.float64)

    log = dict(fingerprint=common.fingerprint(args, model_info, read_metadata(args.data), dict(
        disease=args.disease, n_train_observed=len(train_rows), training_selection=selection_mode,
        n_validation_observed=len(val_rows), n_test=len(test_rows), n_test_observed=int(len(observed['test'])),
        train_patient_ids_sha256=common.sha256_of(splits['train'].patient_id[train_rows]),
        gate=gate, gate_identity=identity, resolved=cfg, warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        betas=list(args.betas))), epochs=[])

    frozen_eligible = common.frozen_embeddings(model, splits['train'], pre_rows, args)
    position = {int(row): i for i, row in enumerate(pre_rows)}
    frozen = {'train': frozen_eligible[[position[int(row)] for row in train_rows]]}
    for name, rows in (('validation', val_rows), ('test', test_rows)):
        frozen[name] = common.frozen_embeddings(model, splits[name], rows, args)
    preprocessing = {}
    for prefix, values in (('embedding', frozen_eligible), ('clinical', clinical('train', pre_rows))):
        for key, value in zip(('median', 'mean', 'sd'), fit_risk_scaler(values)):
            preprocessing[f'{prefix}_{key}'] = value
    del frozen_eligible
    scale = {k: torch.from_numpy(v).to(args.device) for k, v in preprocessing.items()}
    log['fingerprint']['preprocessing_train_patient_ids_sha256'] = common.sha256_of(splits['train'].patient_id[pre_rows])
    log['fingerprint']['n_preprocessing_train_eligible'] = len(pre_rows)
    common.set_seed(args.seed)  # adapters follow the experiment seed, independently of base loading
    snapshot, wrapped = common.prepare_adapters(model, args)
    common.set_seed(args.seed)  # isolate head initialization from model/adapter RNG
    head = torch.nn.Linear(model.config.n_embd + 4, 1).to(args.device)
    y_train, w_train = outcome('train', train_rows)
    raw_y, raw_w = splits['train'].outcomes[args.disease]
    prevalence = float(np.average(raw_y[train_rows].astype(np.float64), weights=raw_w[train_rows].astype(np.float64)))
    common.init_risk_head(head, prevalence)
    mean_weight = float(w_train.mean())
    optimizer, set_warmup = common.make_optimizer(model, head, args, cfg['lr_head'], cfg['weight_decay'])
    accumulator = common.Accumulator(optimizer, set_warmup, args)
    clinical_t = {name: torch.from_numpy(clinical(name, rows)) for name, rows in
                  (('train', train_rows), ('validation', val_rows), ('test', test_rows))}
    state_config = dict(protocol_version=common.PROTOCOL_VERSION, model_sha256=model_info['model_sha256'],
                        bf16=args.bf16, smoke_seed=args.smoke_seed, smoke=bool(model_info['smoke']), recipe=args.recipe, task='risk', arm=args.arm, rank=args.rank, alpha=args.alpha, targets=args.targets,
                        head_in=model.config.n_embd + 4, head_out=1, disease=args.disease)


    def features(z, c):
        return risk_features(z, c, scale)


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
                outputs.append(risk_probabilities(head, z, c, scale).cpu())
        return torch.cat(outputs).numpy().astype(np.float64)


    def validate():
        p = probabilities('validation', val_rows, frozen['validation'] if args.arm == 'head_only' else None)
        y_all, w_all = splits['validation'].outcomes[args.disease]
        y, w = y_all[val_rows].astype(np.float64), w_all[val_rows].astype(np.float64)
        bce = float((w * -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))).sum() / w.sum())
        return bce, weighted_auroc(y, p, w), weighted_brier(y, p, w)


    best, best_state, since_best = None, None, 0
    rng = np.random.default_rng(args.seed)
    candidates, candidate_states, selected_c = [], [], None
    if args.arm == 'head_only':
        x = features(torch.from_numpy(frozen['train']).to(args.device), clinical_t['train'].to(args.device))
        y, w = y_train.to(args.device), w_train.to(args.device)
        for c_value in args.risk_c_values:
            common.set_seed(0)
            fit_risk_candidate(head, x, y, w, prevalence, c_value, cfg['steps'], cfg['lr_head'], args.betas)
            accumulator.steps += cfg['steps']
            bce, auroc, brier = validate()
            candidates.append(dict(C=c_value, weight_decay=1e-4 / c_value,
                                   validation_weighted_bce=bce, validation_weighted_auroc=auroc,
                                   validation_weighted_brier=brier))
            candidate_states.append(common.trainable_state(model, head, preprocessing,
                                    dict(state_config, selected_epoch=1, selected_C=c_value)))
        selected = select_risk_candidate(candidates)
        chosen = candidates[selected]
        selected_c = chosen['C']
        best, best_state = (chosen['validation_weighted_bce'], 1), candidate_states[selected]
        log['candidates'] = candidates
        log['epochs'].append(dict(epoch=1, **chosen, optimizer_steps=accumulator.steps))
    else:
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            head.train()
            losses = []
            for batch_rows in iterate_batches(rng.permutation(len(train_rows)), args.batch_size):
                idx, age, typ = splits['train'].batch(train_rows[batch_rows], args.device)
                with common.autocast(args):
                    z = common.last_hidden(model, idx, age, typ).float()
                logits = head(features(z, clinical_t['train'][batch_rows].to(args.device))).squeeze(1)
                # Fixed training-set mean weight as the normaliser: accumulated over
                # patients this is the gradient of sum(w * BCE) / sum(w), not a
                # per-batch reweighting (PROTOCOL.md Section 5).
                loss = common.weighted_minibatch_loss(logits, y_train[batch_rows].to(args.device),
                                                      w_train[batch_rows].to(args.device), mean_weight)
                accumulator.backward(loss, len(batch_rows))
                losses.append((float(loss.detach()), len(batch_rows)))
            accumulator.end_epoch()
            bce, auroc, brier = validate()
            log['epochs'].append(dict(epoch=epoch, train_weighted_bce=float(np.average([v for v, n in losses], weights=[n for v, n in losses])), validation_weighted_bce=bce,
                                      validation_weighted_auroc=auroc, validation_weighted_brier=brier,
                                      optimizer_steps=accumulator.steps))
            print(json.dumps(log['epochs'][-1]), flush=True)
            if best is None or bce < best[0]:
                best, since_best = (bce, epoch), 0
                best_state = common.trainable_state(model, head, preprocessing, dict(state_config, selected_epoch=epoch))
            else:
                since_best += 1
                if since_best >= cfg['patience']:
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
    summary = dict(arm=args.arm, disease=args.disease, selected=dict(epoch=best[1], validation_weighted_bce=best[0], C=selected_c),
                   test=dict(weighted_auroc=weighted_auroc(y_obs, p_obs, w_obs), weighted_brier=weighted_brier(y_obs, p_obs, w_obs),
                             observed=int(mask.sum()), cases=int((y_obs == 1).sum()), mean_probability=float(p_obs.mean())),
                   optimizer_steps=accumulator.steps,
                   lora=dict(wrapped_modules=wrapped, rank=args.rank, alpha=args.alpha, targets=args.targets) if wrapped else None,
                   audit=common.finish_audit(model, snapshot, head, best_state, args.out))
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


if __name__ == '__main__':
    main()
