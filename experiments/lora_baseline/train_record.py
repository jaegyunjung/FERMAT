"""Task R: diagnoses, prescriptions and procedures recorded on the first clinical
date after the observed history (manuscript Section 2.3), with the frozen
representation arm and the LoRA fine-tuning arm trained in one pipeline.

head_only: frozen last-event hidden state -> standardised -> linear multilabel
           head, trained on cached embeddings with the manuscript recipe.
lora:      LoRA adapters in the transformer and the same linear head trained
           jointly end to end on the same patients and targets. Requires a
           passed reproduction gate (--gate) from a head_only run.
Selection uses validation Recall@10 with validation BCE as tie-breaker; the
test split is scored once with the selected state.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from data import PrefixSplit, class_lookup, iterate_batches, multi_hot, read_metadata  # noqa: E402
from metrics import record_metrics, paired_bootstrap, mean_difference, rank_classes  # noqa: E402

from recipes import configure_recipe, record_epoch_order, record_improved
from tracking import Tracker, add_arguments as add_tracking_arguments
from runtime_guard import RuntimeGuard


def load_diagnostic_embeddings(root, splits, rows, width):
    from prepare_record_v1 import sha256
    result, hashes = {}, {}
    for name, split in splits.items():
        folder = Path(root) / name
        if not np.array_equal(np.load(folder / 'patient_id.npy'), split.patient_id):
            raise ValueError('Diagnostic cache patient order mismatch')
        path = folder / 'frozen_embeddings.npy'
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (len(split), width) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid diagnostic embedding cache')
        result[name], hashes[name] = np.asarray(values[rows[name]]), sha256(path)
    return result, hashes

def main():
    parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
    add_tracking_arguments(parser)
    parser.add_argument('--embeddings-cache', type=Path, help='Saved split caches, diagnostic_v1 only; never issues a real gate')
    parser.add_argument('--reference-embeddings', type=Path, help='Audit freshly extracted v1 embeddings against saved references before fitting')
    parser.add_argument('--gpu-profile', type=Path, help='Enable WSL host/GPU guard with current-boot profile JSON')
    parser.add_argument('--classes', type=Path, default=None,
                        help='int64 .npy of predicted token IDs; defaults to <data>/classes.npy, then the release head classes')
    parser.add_argument('--baseline-recall', type=Path, default=None,
                        help='CSV with patient_id,recall_at_10 from a saved arm for the paired bootstrap')
    parser.add_argument('--baseline-label', default='baseline')
    args = parser.parse_args()
    cfg = common.resolve_defaults(args, dict(
        head_only=dict(epochs=60, patience=10, lr_head=1e-3, weight_decay=0.01, batch=256),
        lora=dict(epochs=20, patience=5, lr_head=1e-3, weight_decay=0.01, batch=args.batch_size)))
    resamples = args.bootstrap_resamples or 5000
    configure_recipe(args, cfg, 'record')
    if args.arm == 'lora' and args.gate is None:
        raise ValueError('LoRA requires --gate from a matching head_only reproduction')
    if (args.embeddings_cache is not None) != (args.recipe == 'diagnostic_v1'):
        raise ValueError('--embeddings-cache requires --recipe diagnostic_v1 and vice versa')
    if args.reference_embeddings is not None and args.recipe != 'v1':
        raise ValueError('--reference-embeddings is for freshly extracted v1 runs only')
    common.set_seed(args.seed)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f'Output directory is not empty: {args.out}; choose a new run directory')
    args.out.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(args.out / 'tracking', args.tracking, args.wandb_project, args.wandb_entity,
                      args.run_name, dict(task='record', arm=args.arm, recipe=args.recipe, seed=args.seed,
                      protocol=common.PROTOCOL_VERSION, commit=common.git_commit(), extraction_batch=args.batch_size,
                      optimizer_batch=cfg['batch'], epochs=cfg['epochs'], patience=cfg['patience'],
                      lr_head=cfg['lr_head'], weight_decay=cfg['weight_decay'], bf16=args.bf16))
    guard = RuntimeGuard(args.gpu_profile, args.out, tracker)
    started = time.perf_counter()

    model, model_info = common.build_model(args)
    splits = {name: PrefixSplit(args.data, name, model.config.block_size) for name in ('train', 'validation', 'test')}
    common.check_split_overlap(splits)
    for name, split in splits.items():
        if split.labels is None:
            raise ValueError(f'{name}: record-task labels are missing')
    if (args.data / 'preprocessing_train').exists():
        pool = PrefixSplit(args.data, 'preprocessing_train', model.config.block_size)
        if not np.isin(splits['train'].patient_id, pool.patient_id).all():
            raise ValueError('Training patients missing from preprocessing pool')
        if any(np.intersect1d(pool.patient_id, splits[s].patient_id).size for s in ('validation', 'test')):
            raise ValueError('Preprocessing pool overlaps held-out patients')
        splits['preprocessing_train'] = pool
    elif args.smoke_random_init is None:
        raise ValueError('Real record runs require the original all-training preprocessing pool')
    classes_path = args.classes or (args.data / 'classes.npy' if (args.data / 'classes.npy').exists()
                                    else args.release / 'artifacts/head_classes.npy')
    classes = np.load(classes_path, allow_pickle=False).astype(np.int64)
    class_index = class_lookup(classes, model.config.vocab_size)
    n_classes = len(classes)
    train_rows, selection_mode = common.select_training_rows(splits['train'], args)
    val_rows, test_rows = np.arange(len(splits['validation'])), np.arange(len(splits['test']))
    label_hash = common.sha256_of_arrays(classes, splits['validation'].label_indptr, splits['validation'].label_tokens,
                                         splits['test'].label_indptr, splits['test'].label_tokens)
    identity = common.gate_identity('record', None, model_info, splits, label_hash, train_rows, selection_mode,
                                    read_metadata(args.data), precision='bf16' if args.bf16 else 'float32')
    gate = common.require_gate(args, identity)
    targets = {name: multi_hot(splits[name].label_lists(rows), class_index, n_classes)
               for name, rows in (('train', train_rows), ('validation', val_rows))}
    log = dict(fingerprint=common.fingerprint(args, model_info, read_metadata(args.data), dict(
        classes=str(classes_path), n_classes=n_classes, n_train=len(train_rows), training_selection=selection_mode,
        n_validation=len(val_rows), n_test=len(test_rows),
        train_patient_ids_sha256=common.sha256_of(splits['train'].patient_id[train_rows]),
        gate=gate, gate_identity=identity, resolved=cfg, warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        betas=list(args.betas))), epochs=[])

    # Standardisation always comes from the frozen model on the training patients,
    # so both arms feed the head inputs on the same scale (MODEL_CARD.md).
    split_rows = dict(train=train_rows, validation=val_rows, test=test_rows)
    if 'preprocessing_train' in splits:
        split_rows = dict(preprocessing_train=np.arange(len(splits['preprocessing_train'])), **split_rows)
    frozen = {}
    if args.embeddings_cache is not None:
        frozen, hashes = load_diagnostic_embeddings(args.embeddings_cache, splits, split_rows, model.config.n_embd)
        log['fingerprint']['diagnostic_cache_sha256'] = hashes
    else:
        for name, rows in split_rows.items():
            if name == 'train' and 'preprocessing_train' in frozen:
                positions = {int(pid): i for i, pid in enumerate(splits['preprocessing_train'].patient_id)}
                frozen['train'] = frozen['preprocessing_train'][[positions[int(pid)] for pid in splits['train'].patient_id[train_rows]]]
                continue
            phase_started, last_log = time.perf_counter(), [0.]
            def progress(done, total):
                guard.check()
                elapsed = time.perf_counter() - phase_started
                if elapsed - last_log[0] >= 10 or done == total:
                    tracker.log({f'progress/embedding_{name}': done / total,
                                 f'time/embedding_{name}_seconds': elapsed,
                                 f'time/embedding_{name}_eta_seconds': elapsed / done * (total - done)})
                    last_log[0] = elapsed
            frozen[name] = common.frozen_embeddings(model, splits[name], rows, args, progress)
    log['fingerprint']['embedding_source'] = 'diagnostic_cache' if args.embeddings_cache else 'release_forward'
    if args.reference_embeddings is not None:
        reference, hashes = load_diagnostic_embeddings(args.reference_embeddings, splits, split_rows, model.config.n_embd)
        parity = {name: dict(passed=bool(np.allclose(frozen[name], reference[name], rtol=1e-5, atol=1e-6)),
                           maximum_absolute_error=float(np.abs(frozen[name] - reference[name]).max()),
                           exact_fraction=float(np.mean(frozen[name] == reference[name]))) for name in split_rows}
        common.write_json(args.out / 'embedding_parity.json', dict(reference_sha256=hashes, splits=parity))
        log['fingerprint']['embedding_parity'] = parity
        if not all(value['passed'] for value in parity.values()):
            raise ValueError('Fresh embeddings differ from the reference; head fitting not started')
        del reference
    log['fingerprint']['embedding_phase_seconds'] = time.perf_counter() - started
    tracker.log({'time/preparation_seconds': time.perf_counter() - started})
    mean, sd = common.standardization(frozen.get('preprocessing_train', frozen['train']))
    log['fingerprint']['n_preprocessing_train'] = len(frozen.get('preprocessing_train', frozen['train']))
    mean_t, sd_t = (torch.from_numpy(a).to(args.device) for a in (mean, sd))
    common.set_seed(args.seed)  # adapters follow the experiment seed, independently of base loading
    snapshot, wrapped = common.prepare_adapters(model, args)
    common.set_seed(args.seed)  # isolate head initialization from model/adapter RNG
    head = torch.nn.Linear(model.config.n_embd, n_classes).to(args.device)
    common.init_record_head(head, targets['train'].sum(0).numpy(), len(train_rows))
    optimizer, set_warmup = common.make_optimizer(model, head, args, cfg['lr_head'], cfg['weight_decay'])
    accumulator = common.Accumulator(optimizer, set_warmup, args)
    state_config = dict(protocol_version=common.PROTOCOL_VERSION, model_sha256=model_info['model_sha256'],
                        bf16=args.bf16, smoke_seed=args.smoke_seed, smoke=bool(model_info['smoke']), recipe=args.recipe, task='record', arm=args.arm, rank=args.rank, alpha=args.alpha, targets=args.targets,
                        head_in=model.config.n_embd, head_out=n_classes, classes=str(classes_path))
    preprocessing = dict(embedding_mean=mean, embedding_sd=sd, classes=classes)


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


    best, best_state, since_best = None, None, 0
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, cfg['epochs'] + 1):
        epoch_started = time.perf_counter()
        guard.check()
        head.train()
        model.train() if args.arm == 'lora' else model.eval()
        order = (record_epoch_order(len(train_rows), args.seed, epoch - 1)
                 if args.recipe in ('v1', 'diagnostic_v1') else rng.permutation(len(train_rows)))
        losses = []
        for batch_rows in iterate_batches(order, cfg['batch']):
            guard.check()
            y = targets['train'][batch_rows].to(args.device)
            if args.arm == 'head_only':
                z = torch.from_numpy(frozen['train'][batch_rows]).to(args.device)
            else:
                idx, age, typ = splits['train'].batch(train_rows[batch_rows], args.device)
                with common.autocast(args):
                    z = common.last_hidden(model, idx, age, typ).float()
            loss = common.record_loss(head((z - mean_t) / sd_t), y)
            accumulator.backward(loss, len(batch_rows))
            losses.append((float(loss.detach()), len(batch_rows)))
        accumulator.end_epoch()
        recall, bce = validate()
        log['epochs'].append(dict(epoch=epoch, train_bce=float(np.average([v for v, n in losses], weights=[n for v, n in losses])), validation_recall_at_10=recall,
                                  validation_bce=bce, optimizer_steps=accumulator.steps))
        print(json.dumps(log['epochs'][-1]), flush=True)
        epoch_seconds = time.perf_counter() - epoch_started
        log['epochs'][-1]['epoch_seconds'] = epoch_seconds
        tracker.log({'progress/epoch': epoch, 'train/bce': log['epochs'][-1]['train_bce'],
                     'validation/recall_at_10': recall, 'validation/bce': bce,
                     'time/epoch_seconds': epoch_seconds,
                     'time/max_remaining_training_seconds': epoch_seconds * (cfg['epochs'] - epoch)})
        common.write_json(args.out / 'progress.json', dict(epoch=epoch, epochs=log['epochs'],
                          remaining_epoch_cap=cfg['epochs']-epoch))
        improved = record_improved(recall, bce, best)
        if improved:
            best, since_best = (recall, bce, epoch), 0
            best_state = common.trainable_state(model, head, preprocessing, dict(state_config, selected_epoch=epoch))
        else:
            since_best += 1
            if since_best >= cfg['patience']:
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
    torch.save(dict(patient_id=torch.from_numpy(patient_ids), logits=test_logits,
                    top10_classes=torch.from_numpy(classes[rank_classes(test_logits, 10).numpy()])),
               args.out / 'test_scores.pt')
    summary = dict(arm=args.arm, selected=dict(epoch=best[2], validation_recall_at_10=best[0], validation_bce=best[1]),
                   test={k: float(v.mean()) for k, v in per_patient.items()},
                   optimizer_steps=accumulator.steps,
                   lora=dict(wrapped_modules=wrapped, rank=args.rank, alpha=args.alpha, targets=args.targets) if wrapped else None,
                   audit=common.finish_audit(model, snapshot, head, best_state, args.out))
    if args.baseline_recall is not None:
        baseline = common.align_baseline(patient_ids, common.read_baseline_csv(args.baseline_recall, 'recall_at_10'),
                                         args.baseline_label)
        summary['paired_recall_at_10_minus_' + args.baseline_label] = paired_bootstrap(
            mean_difference, [per_patient['recall_at_10'], baseline], resamples, args.seed)
    log['summary'] = summary
    summary['elapsed_seconds'] = time.perf_counter() - started
    common.write_json(args.out / 'summary.json', log)
    tracker.log({'test/' + k: v for k, v in summary['test'].items()})
    if args.recipe == 'v1':
        from gates import make_gate, official_targets
        result = make_gate(log, official_targets('record', None))
        common.write_json(args.out / 'gate.json', result)
        tracker.log({'gate/passed': result['passed']})
    guard.check(force=True)
    tracker.finish()
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
