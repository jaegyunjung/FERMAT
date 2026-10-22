"""Task-specific v1 recipes, kept separate from LoRA optimization settings.

Sources: run_next_record_bundle.py:fit, train_task30_d3_comparators.py,
task30_d3_analysis_core.py:fit_transformer. See PROTOCOL.md for claim limits.
"""
import warnings
import numpy as np
import torch

RECORD_SEED = 20260908
RISK_C_VALUES = (0.01, 0.1, 1.0, 10.0)
V1_RISK_C = 10.0


def risk_c_values(recipe, values=None, steps=200):
    """The published revision fixed C=10; the earlier grid is exploratory."""
    values = tuple(values) if values is not None else ((V1_RISK_C,) if recipe == 'v1' else RISK_C_VALUES)
    if steps <= 0 or not values or any(not np.isfinite(c) or c <= 0 for c in values):
        raise ValueError('Risk steps and C values must be positive and finite')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate risk C values')
    if recipe == 'v1' and (values != (V1_RISK_C,) or steps != 200):
        raise ValueError('v1 requires fixed C=10 and 200 steps; use --recipe experimental for a grid')
    return values


def record_epoch_order(n, seed, epoch):
    """epoch is zero-based, as in the original fit function."""
    epoch_seed = seed + epoch * 101
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(epoch_seed)
    return np.random.default_rng(epoch_seed).permutation(n)


def record_improved(score, loss, best):
    return best is None or score > best[0] + 1e-10 or (abs(score - best[0]) <= 1e-10 and loss < best[1])


def fit_risk_scaler(values):
    """Fit on all eligible training patients, including censored patients."""
    # The original pandas transformer reduces contiguous columns. Preserve that
    # summation order after NumPy row selection, which otherwise makes a C array.
    x = np.asfortranarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('Risk preprocessing needs a nonempty eligible training cohort')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-missing column -> zero
        median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    x = np.where(np.isfinite(x), x, median)
    mean, sd = x.mean(0), x.std(0)
    return median, mean, np.where(sd > 1e-8, sd, 1.)


def risk_features(embedding, clinical, pre, dtype=torch.float32):
    """Shared training/prediction transform; float64 arithmetic, float32 head.

    Clinical columns precede embedding columns, matching the original predictor.
    Torch operations retain gradients from the head into the LoRA embedding.
    """
    parts = []
    for name, values in (('clinical', clinical), ('embedding', embedding)):
        values = values.to(torch.float64)
        median, mean, sd = (pre[f'{name}_{key}'].to(values.device) for key in ('median', 'mean', 'sd'))
        values = torch.where(torch.isfinite(values), values, median)
        parts.append(((values - mean) / sd).to(dtype))
    return torch.cat(parts, dim=1)


def risk_probabilities(head, embedding, clinical, pre):
    """v1 evaluates the fitted FP32 coefficients in float64, clipping logits."""
    x = risk_features(embedding, clinical, pre, dtype=torch.float64)
    logits = torch.nn.functional.linear(x, head.weight.double(), head.bias.double()).squeeze(1)
    return torch.sigmoid(logits.clamp(-40, 40))


def select_risk_candidate(candidates):
    if not candidates or not all(np.isfinite(c['validation_weighted_brier']) for c in candidates):
        raise ValueError('Risk selection requires finite validation Brier scores')
    return min(range(len(candidates)), key=lambda i: (candidates[i]['validation_weighted_brier'], candidates[i]['C']))


def fit_risk_candidate(head, x, y, weights, prevalence, c_value, steps=200, lr=.03, betas=(.9, .999)):
    """One original full-batch fit; validation and C selection stay outside."""
    from common import init_risk_head, weighted_objective
    init_risk_head(head, prevalence)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4 / c_value, betas=tuple(betas))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_objective(head(x).squeeze(1), y, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite risk training loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise FloatingPointError('Nonfinite risk training gradient')
        optimizer.step()
    return float(loss.detach())


def configure_recipe(args, cfg, task):
    """Reject silent deviations from v1; exploratory runs cannot issue a real gate."""
    args.recipe = args.recipe or ('v1' if args.arm == 'head_only' else 'lora')
    if (args.arm == 'lora') != (args.recipe == 'lora'):
        raise ValueError('LoRA arm requires recipe=lora; head_only requires v1 or experimental')
    if args.recipe == 'diagnostic_v1' and task != 'record':
        raise ValueError('diagnostic_v1 is only implemented for the record task')
    if args.seed is None:
        args.seed = (RECORD_SEED if task == 'record' else 0) if args.recipe in ('v1', 'diagnostic_v1') else 1
    if args.recipe not in ('v1', 'diagnostic_v1') or args.smoke_random_init is not None:
        return
    if args.grad_accum != 1 or args.grad_clip is not None or args.warmup_steps != 0 or tuple(args.betas) != (0.9, 0.999):
        raise ValueError('v1 requires accumulation=1, no clipping/warm-up and default AdamW betas; use --recipe experimental')
    expected = dict(lr_head=.001, weight_decay=.01, epochs=60, patience=10) if task == 'record' else dict(lr_head=.03, weight_decay=1e-5)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('Changed v1 hyperparameters; use --recipe experimental')
    if task == 'record' and args.seed not in (RECORD_SEED, RECORD_SEED + 1, RECORD_SEED + 2):
        raise ValueError('v1 record seed must be one of 20260908, 20260909, 20260910')
