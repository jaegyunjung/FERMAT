"""Task D: five-year disease risk for one disease (manuscript Section 2.2 detailed
comparison), with the frozen representation arm and the LoRA arm in one pipeline.

head_only: frozen hidden state and the four clinical variables -> weighted
           logistic predictor fitted with the manuscript recipe (AdamW, 200
           full-batch steps, learning rate 0.03, fixed C=10).
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
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from data import PrefixSplit, iterate_batches, read_metadata  # noqa: E402
from metrics import weighted_auroc, weighted_brier, paired_bootstrap  # noqa: E402

from recipes import (configure_recipe, fit_risk_scaler, risk_features, risk_probabilities,
                     select_risk_candidate, fit_risk_candidate, risk_c_values)
from tracking import Tracker, add_arguments as add_tracking_arguments
from runtime_guard import RuntimeGuard
import checkpointing

def main():
    parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
    add_tracking_arguments(parser)
    parser.add_argument('--embedding-cache', type=Path, help='Verified original-batch CUDA risk cache')
    parser.add_argument('--reference-model', type=Path, help='Original C10 JSON: exact preprocessing and saved-head replay audit before v1 fit')
    parser.add_argument('--gpu-profile', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--matched-head', action='store_true', help='Experimental mini-batch frozen-head schedule control')
    parser.add_argument('--benchmark-steps', type=int)
    parser.add_argument('--benchmark-warmup', type=int, default=10)
    parser.add_argument('--disease', required=True, help='Suffix of outcome_<disease>.npy / weight_<disease>.npy')
    parser.add_argument('--baseline-probabilities', type=Path, default=None,
                        help='CSV with patient_id,probability from a saved arm for the paired bootstrap')
    parser.add_argument('--baseline-label', default='baseline')
    parser.add_argument('--risk-c-values', type=float, nargs='+', default=None,
                        help='v1: fixed 10; experimental: default .01 .1 1 10, selected by validation Brier')
    parser.add_argument('--risk-steps', type=int, default=200)
    args = parser.parse_args()
    full_batch = args.arm == 'head_only' and not args.matched_head
    if args.matched_head and (args.arm != 'head_only' or args.recipe != 'experimental'):
        parser.error('--matched-head requires experimental head_only')
    if full_batch and args.weight_decay is not None:
        parser.error('Risk head_only derives weight decay from --risk-c-values; do not pass --weight-decay')
    if full_batch and (args.grad_clip is not None or args.warmup_steps not in (None, 0)):
        parser.error('Full-batch risk head_only does not use clipping or warm-up')
    if args.matched_head:
        args.warmup_steps = 100 if args.warmup_steps is None else args.warmup_steps
        args.grad_clip = 1. if args.grad_clip is None else args.grad_clip
    cfg = common.resolve_defaults(args, dict(
        head_only=(dict(epochs=3, patience=1, lr_head=.001, weight_decay=.01) if args.matched_head else
                   dict(epochs=1, patience=1, lr_head=0.03, weight_decay=1e-5, steps=args.risk_steps)),
        lora=dict(epochs=3, patience=1, lr_head=1e-3, weight_decay=0.01)))
    resamples = args.bootstrap_resamples or 2000
    configure_recipe(args, cfg, 'risk')
    if full_batch:
        args.risk_c_values = risk_c_values(args.recipe, args.risk_c_values, args.risk_steps)
    elif args.risk_c_values is not None or args.risk_steps != 200:
        raise ValueError('Risk C candidates and full-batch steps only apply to head_only')
    if args.validation_only and args.recipe == 'v1':
        raise ValueError('Validation-only cannot issue a v1 gate')
    if args.reference_model and args.recipe != 'v1':
        raise ValueError('Reference model audit is restricted to v1 reproduction')
    if not args.smoke_random_init and args.recipe == 'v1' and args.reference_model is None:
        raise ValueError('Real v1 reproduction requires the original --reference-model')
    if (args.resume or args.stop_after_epoch) and full_batch:
        raise ValueError('Epoch resume/pause requires a minibatch arm')
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError('Stop epoch must be positive')
    if args.benchmark_steps is not None and (args.arm != 'lora' or args.grad_accum != 1
            or args.resume or args.stop_after_epoch or args.validation_only
            or args.benchmark_warmup < 0 or args.benchmark_steps <= args.benchmark_warmup):
        raise ValueError('Benchmark requires LoRA, accumulation 1 and steps > warmup; no resume/pause/validation-only')
    common.set_seed(args.seed)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.out}; choose a new run directory')
    args.out.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(args.out / 'tracking', args.tracking, args.wandb_project, args.wandb_entity, args.run_name,
                      dict(task='risk', disease=args.disease, arm=args.arm, recipe=args.recipe, seed=args.seed,
                           matched_head=args.matched_head, validation_only=args.validation_only,
                           resumed=args.resume is not None, benchmark_steps=args.benchmark_steps,
                           batch=args.batch_size, epochs=cfg['epochs'], patience=cfg['patience'],
                           lr_head=cfg['lr_head'], lr_lora=args.lr_lora, bf16=args.bf16,
                           rank=args.rank, alpha=args.alpha, lora_dropout=args.lora_dropout,
                           weight_decay=cfg['weight_decay'], warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
                           commit=common.git_commit(), protocol=common.PROTOCOL_VERSION))
    guard = RuntimeGuard(args.gpu_profile, args.out, tracker)
    started = time.perf_counter()

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
    if full_batch and args.grad_accum != 1:
        raise ValueError('--grad-accum does not apply to the full-batch head_only fit')
    label_hash = common.sha256_of_arrays(*splits['validation'].outcomes[args.disease], *splits['test'].outcomes[args.disease])
    metadata = read_metadata(args.data)
    cached, cache_hash = None, None
    if args.embedding_cache:
        from risk_cache import load_cache
        cached, cache_hash = load_cache(args.embedding_cache, args.data, model_info, args, splits)
        metadata = dict(metadata, verified_embedding_cache_sha256=cache_hash)
    elif not model_info['smoke']:
        raise ValueError('Real disease runs require a verified original-batch embedding cache')
    identity = common.gate_identity('risk', args.disease, model_info, splits, label_hash, train_rows, selection_mode,
                                    metadata, precision='bf16' if args.bf16 else 'float32')
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
        betas=list(args.betas),
        risk_regularization=(dict(selection='fixed' if args.recipe == 'v1' else 'validation_brier',
                                  candidates=list(args.risk_c_values), steps_per_fit=args.risk_steps)
                             if full_batch else None), verified_embedding_cache_sha256=cache_hash)), epochs=[])

    frozen_eligible = (cached['train'][pre_rows] if cached is not None else
                       common.frozen_embeddings(model, splits['train'], pre_rows, args))
    position = {int(row): i for i, row in enumerate(pre_rows)}
    frozen = {'train': frozen_eligible[[position[int(row)] for row in train_rows]]}
    for name, rows in (('validation', val_rows), ('test', test_rows)):
        frozen[name] = cached[name][rows] if cached is not None else common.frozen_embeddings(model, splits[name], rows, args)
    del cached
    preprocessing = {}
    for prefix, values in (('embedding', frozen_eligible), ('clinical', clinical('train', pre_rows))):
        for key, value in zip(('median', 'mean', 'sd'), fit_risk_scaler(values)):
            preprocessing[f'{prefix}_{key}'] = value
    del frozen_eligible
    scale = {k: torch.from_numpy(v).to(args.device) for k, v in preprocessing.items()}
    if args.reference_model:
        reference = json.loads(args.reference_model.read_text())
        from prepare_risk_baselines import DISEASES
        if reference['fixed_C'] != 10 or reference['phenotype'] != DISEASES[args.disease]:
            raise ValueError('Reference model disease or C differs')
        expected = reference['transformer']
        for prefix, sl in (('clinical', slice(0, 4)), ('embedding', slice(4, None))):
            for key, old in (('median', 'medians'), ('mean', 'means'), ('sd', 'scales')):
                if not np.array_equal(preprocessing[f'{prefix}_{key}'], np.asarray(expected[old])[sl]):
                    raise ValueError(f'Original preprocessing differs: {prefix}_{key}')
        ref_head = torch.nn.Linear(model.config.n_embd+4, 1).to(args.device)
        with torch.no_grad():
            raw_features = np.concatenate([clinical('test', test_rows), frozen['test'].astype(np.float64)], axis=1)
            clean = np.where(np.isfinite(raw_features), raw_features, np.asarray(expected['medians']))
            expected_features = ((clean-np.asarray(expected['means']))/np.asarray(expected['scales'])).astype(np.float32)
            actual_features = risk_features(torch.from_numpy(frozen['test']).to(args.device),
                torch.from_numpy(clinical('test', test_rows)).to(args.device), scale).cpu().numpy()
            if not np.array_equal(actual_features, expected_features):
                raise ValueError('Original FP32 features differ on the execution device')
            ref_head.weight.copy_(torch.tensor(reference['coefficients'], device=args.device).reshape(1, -1))
            ref_head.bias.fill_(reference['intercept'])
            replay = risk_probabilities(ref_head, torch.from_numpy(frozen['test']).to(args.device),
                torch.from_numpy(clinical('test', test_rows)).to(args.device), scale).cpu().numpy()
        saved = np.load(args.data / 'test' / f'baseline_probability_{args.disease}.npy')
        error = float(np.max(np.abs(replay-saved)))
        if error > 1e-6:
            raise ValueError('Original C10 probability replay differs')
        log['fingerprint']['reference_audit'] = dict(preprocessing_exact=True, fp32_features_exact=True, maximum_probability_error=error,
            model_sha256=checkpointing.sha256(args.reference_model))
        del ref_head
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
    state_config.update(validation_only=args.validation_only, inference_batch_size=args.batch_size,
                        risk_validation_observed=True,
                        validation_patient_ids_sha256=common.sha256_of(splits['validation'].patient_id[val_rows]))
    tracker.log({'time/preparation_seconds': time.perf_counter()-started})


    def features(z, c):
        return risk_features(z, c, scale)


    def probabilities(name, rows, cached):
        outputs = []
        model.eval()
        with torch.no_grad():
            for batch_rows in iterate_batches(np.arange(len(rows)), args.batch_size):
                guard.check()
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


    def train_batch(batch_rows):
        if args.arm == 'head_only':
            z = torch.from_numpy(frozen['train'][batch_rows]).to(args.device)
        else:
            idx, age, typ = splits['train'].batch(train_rows[batch_rows], args.device)
            with common.autocast(args):
                z = common.last_hidden(model, idx, age, typ).float()
        logits = head(features(z, clinical_t['train'][batch_rows].to(args.device))).squeeze(1)
        loss = common.weighted_minibatch_loss(logits, y_train[batch_rows].to(args.device),
                                             w_train[batch_rows].to(args.device), mean_weight)
        accumulator.backward(loss, len(batch_rows))
        return float(loss.detach())

    if args.benchmark_steps is not None:
        from benchmarking import measure_training, estimate_runtime
        before = common.lora_state_dict(model)
        model.train(); head.train()
        report = measure_training(np.random.default_rng(args.seed).permutation(len(train_rows)), args.batch_size,
            args.benchmark_steps, args.benchmark_warmup, train_batch, validate, guard, tracker, args.device,
            len(val_rows), len(test_rows), time.perf_counter()-started)
        report['eta'] = estimate_runtime(report['step_seconds'][args.benchmark_warmup:], args.batch_size,
            len(train_rows), report['validation_seconds'], time.perf_counter()-started-report['training_seconds']-report['validation_seconds'],
            len(val_rows), len(test_rows), epochs=(1, 2, 3))
        state = common.trainable_state(model, head, preprocessing, dict(state_config, benchmark_only=True, selected_epoch=0))
        report['adapter_tensors_changed'] = sum(not torch.equal(v, before[k]) for k,v in state['lora'].items())
        if not report['adapter_tensors_changed']:
            raise ValueError('Benchmark did not update adapters')
        report['audit'] = common.finish_audit(model, snapshot, head, state, args.out)
        report['optimizer_steps'] = accumulator.steps
        report['elapsed_seconds'] = time.perf_counter()-started
        common.write_json(args.out / 'benchmark.json', dict(fingerprint=log['fingerprint'], benchmark=report))
        tracker.log({'progress/benchmark_completed': True, 'audit/frozen_base_unchanged': True})
        tracker.finish()
        return

    best, best_state, since_best = None, None, 0
    rng = np.random.default_rng(args.seed)
    candidates, candidate_states, selected_c = [], [], None
    if full_batch:
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
        selected = 0 if args.recipe == 'v1' else select_risk_candidate(candidates)
        chosen = candidates[selected]
        selected_c = chosen['C']
        best, best_state = (chosen['validation_weighted_bce'], 1), candidate_states[selected]
        log['candidates'] = candidates
        log['epochs'].append(dict(epoch=1, **chosen, optimizer_steps=accumulator.steps))
    else:
        contract = checkpointing.training_contract(args, cfg, identity, cache_hash)
        completed_epoch = 0
        if args.resume:
            previous, manifest = checkpointing.load_epoch(args.resume, contract)
            for key, value in preprocessing.items():
                if not torch.equal(torch.as_tensor(value), previous['current_state']['preprocessing'][key]):
                    raise ValueError('Resume preprocessing mismatch: ' + key)
            common.restore_state(model, head, previous['current_state'])
            optimizer.load_state_dict(previous['optimizer'])
            accumulator.steps = accumulator.epoch_start = previous['optimizer_steps']
            best, best_state, since_best = previous['best'], previous['best_state'], previous['since_best']
            completed_epoch, log['epochs'] = previous['completed_epoch'], previous['epochs']
            log['fingerprint']['resume'] = dict(checkpoint_sha256=manifest['sha256'], completed_epoch=completed_epoch,
                                               source_run_id=previous.get('tracking_run_id'))
            checkpointing.atomic_torch_save(best_state, args.out / 'best_model.pt')
            tracker.log({'progress/resumed_after_epoch': completed_epoch})
            # Restore last, after model/cache/optimizer/tracker construction.
            checkpointing.restore_rng(previous['rng'], rng)
        terminal = completed_epoch >= cfg['epochs'] or since_best >= cfg['patience']
        for epoch in range(completed_epoch + 1, cfg['epochs'] + 1) if not terminal else ():
            epoch_started = time.perf_counter()
            model.train() if args.arm == 'lora' else model.eval()
            head.train()
            losses = []
            for batch_rows in iterate_batches(rng.permutation(len(train_rows)), args.batch_size):
                guard.check()
                losses.append((train_batch(batch_rows), len(batch_rows)))
            accumulator.end_epoch()
            bce, auroc, brier = validate()
            log['epochs'].append(dict(epoch=epoch, train_weighted_bce=float(np.average([v for v, n in losses], weights=[n for v, n in losses])), validation_weighted_bce=bce,
                                      validation_weighted_auroc=auroc, validation_weighted_brier=brier,
                                      optimizer_steps=accumulator.steps))
            log['epochs'][-1]['epoch_seconds'] = time.perf_counter()-epoch_started
            print(json.dumps(log['epochs'][-1]), flush=True)
            improved = best is None or bce < best[0]
            if improved:
                best, since_best = (bce, epoch), 0
                best_state = common.trainable_state(model, head, preprocessing, dict(state_config, selected_epoch=epoch))
            else:
                since_best += 1
            tracker.log({'progress/epoch': epoch, 'progress/optimizer_steps': accumulator.steps,
                         'train/bce': log['epochs'][-1]['train_weighted_bce'],
                         'validation/bce': bce, 'validation/auroc': auroc, 'validation/brier': brier,
                         'time/epoch_seconds': log['epochs'][-1]['epoch_seconds']})
            common.write_json(args.out / 'progress.json', dict(epoch=epoch, epochs=log['epochs']))
            payload = dict(contract=contract, completed_epoch=epoch, optimizer_steps=accumulator.steps,
                current_state=common.trainable_state(model, head, preprocessing, dict(state_config, selected_epoch=epoch)),
                optimizer=checkpointing.cpu_tree(optimizer.state_dict()), best=best, best_state=best_state,
                since_best=since_best, epochs=log['epochs'], rng=checkpointing.capture_rng(rng),
                git_commit=log['fingerprint']['git_commit'], tracking_run_id=tracker.status.get('run_id'))
            manifest = checkpointing.save_epoch(args.out, payload, improved)
            terminal = epoch >= cfg['epochs'] or since_best >= cfg['patience']
            if args.stop_after_epoch and epoch >= args.stop_after_epoch and not terminal:
                common.write_json(args.out / 'paused.json', dict(completed_epoch=epoch, manifest=manifest, test_evaluated=False))
                guard.check(force=True)
                tracker.finish()
                return
            if terminal:
                break

    common.restore_state(model, head, best_state)
    if args.validation_only:
        p = probabilities('validation', val_rows, frozen['validation'] if args.arm == 'head_only' else None)
        checkpointing.atomic_torch_save(dict(patient_id=torch.from_numpy(splits['validation'].patient_id[val_rows].copy()),
            probabilities=torch.from_numpy(p)), args.out / 'validation_scores.pt')
        audit = common.finish_audit(model, snapshot, head, best_state, args.out)
        log['summary'] = dict(validation_only=True, test_evaluated=False, selected_epoch=best[1], audit=audit,
                              optimizer_steps=accumulator.steps)
        common.write_json(args.out / 'validation_summary.json', log)
        guard.check(force=True)
        tracker.finish()
        return
    p_test = probabilities('test', test_rows, frozen['test'] if args.arm == 'head_only' else None)
    y_all, w_all = splits['test'].outcomes[args.disease]
    patient_ids = splits['test'].patient_id[test_rows]
    checkpointing.atomic_torch_save(dict(patient_id=torch.from_numpy(patient_ids.copy()), probabilities=torch.from_numpy(p_test)),
                                   args.out / 'test_scores.pt')
    with (args.out / 'test_patient_probabilities.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patient_id', 'probability', 'outcome', 'weight'])
        for row in range(len(patient_ids)):
            writer.writerow([int(patient_ids[row]), f'{p_test[row]:.17g}', int(y_all[row]), f'{w_all[row]:.17g}'])
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
    summary['elapsed_seconds'] = time.perf_counter()-started
    common.write_json(args.out / 'summary.json', log)
    tracker.log({'test/' + k: v for k, v in summary['test'].items()})
    if args.recipe == 'v1':
        from gates import make_gate, official_targets
        result = make_gate(log, official_targets('risk', args.disease))
        common.write_json(args.out / 'gate.json', result)
        tracker.log({'gate/passed': result['passed']})
    guard.check(force=True)
    tracker.finish()
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
