"""Bounded LoRA training timing; no test scoring or reproduction gate issuance."""
import math
import time
import numpy as np
import torch


def estimate_runtime(step_seconds, batch_size, n_train, validation_seconds,
                     preparation_seconds, n_validation, n_test, epochs=(5, 10, 20)):
    values = np.asarray(step_seconds, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Positive finite measured step durations required')
    batches = math.ceil(n_train / batch_size)
    training = float(values.mean()) * batches
    evaluation = validation_seconds * n_test / n_validation
    return dict(batches_per_epoch=batches, training_seconds_per_epoch=training,
                validation_seconds_per_epoch=validation_seconds,
                seconds_per_epoch=training + validation_seconds,
                test_seconds_estimated=evaluation,
                scenarios=[dict(epochs=n, one_seed_seconds=preparation_seconds + n * (training + validation_seconds) + evaluation,
                                three_seeds_seconds=3 * (preparation_seconds + n * (training + validation_seconds) + evaluation))
                           for n in epochs],
                exclusions='Checkpoint/serialization, bootstrap, network finish and other GPU contention; use 15-25% scheduling reserve. No early-stop prediction.')


def measure_training(order, batch_size, steps, warmup, train_batch, validate, guard,
                     tracker, device, n_validation, n_test, preparation_seconds):
    if steps <= warmup or steps > math.ceil(len(order) / batch_size):
        raise ValueError('Benchmark needs more than warmup steps and at most one training epoch')
    cuda = str(device).startswith('cuda')
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    timings, losses, patients = [], [], 0
    for step in range(steps):
        rows = order[step * batch_size:(step + 1) * batch_size]
        sync()
        started = time.perf_counter()
        guard.check()
        loss = train_batch(rows)
        sync()
        duration = time.perf_counter() - started
        timings.append(duration)
        losses.append(loss)
        patients += len(rows)
        tracker.log({'progress/benchmark_step': step + 1, 'train/bce': loss,
                     'time/train_step_seconds': duration,
                     'time/benchmark_train_eta_seconds': float(np.mean(timings[-10:])) * (steps - step - 1)})
    training_peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                         reserved_bytes=torch.cuda.max_memory_reserved(device)) if cuda else None
    sync()
    started = time.perf_counter()
    validate()  # exercise the full validation path; discard exploratory accuracy
    sync()
    validation_seconds = time.perf_counter() - started
    guard.check(force=True)
    steady = timings[warmup:]
    eta = estimate_runtime(steady, batch_size, len(order), validation_seconds,
                           preparation_seconds, n_validation, n_test)
    peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                reserved_bytes=torch.cuda.max_memory_reserved(device)) if cuda else None
    tracker.log({'time/steady_step_mean_seconds': float(np.mean(steady)),
                 'time/full_validation_seconds': validation_seconds,
                 'time/full_epoch_estimated_seconds': eta['seconds_per_epoch'],
                 **({'gpu/peak_allocated_mib': peak['allocated_bytes'] / 2**20,
                     'gpu/peak_reserved_mib': peak['reserved_bytes'] / 2**20} if cuda else {})})
    return dict(benchmark_only=True, test_evaluated=False, steps=steps,
                patients_processed=patients, batch_size=batch_size,
                excluded_startup_steps=warmup, step_seconds=timings,
                steady_step_mean_seconds=float(np.mean(steady)),
                steady_step_median_seconds=float(np.median(steady)),
                steady_step_p10_p90_seconds=np.quantile(steady, [.1, .9]).tolist(),
                steady_patients_per_second=sum(len(order[i*batch_size:(i+1)*batch_size]) for i in range(warmup, steps)) / sum(steady),
                training_seconds=sum(timings), validation_seconds=validation_seconds,
                first_training_loss=losses[0], last_training_loss=losses[-1],
                training_peak_cuda_memory=training_peak, overall_peak_cuda_memory=peak, eta=eta)
