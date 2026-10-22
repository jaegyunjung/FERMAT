"""Patient-level metrics matching the manuscript definitions, plus paired bootstraps."""
import numpy as np
import torch

RECORD_KS = (1, 2, 5, 10)


def rank_classes(scores, k):
    """Top-k columns per row with stable, outcome-independent tie order."""
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]


def record_metrics(scores, label_lists, class_index, ks=RECORD_KS):
    """Per-patient Recall@K, F1@K, Precision@K and Hit@K (percent), as in Section 2.3."""
    top = rank_classes(scores, max(ks)).cpu().numpy()
    out = {f'{name}_at_{k}': np.zeros(len(label_lists)) for name in ('recall', 'f1', 'precision', 'hit') for k in ks}
    for row, tokens in enumerate(label_lists):
        truth = set(class_index[np.asarray(tokens, dtype=np.int64)].tolist())
        truth.discard(-1)
        if not truth:
            raise ValueError('Observed target set is empty inside the prediction vocabulary')
        for k in ks:
            hits = len(truth.intersection(top[row, :k].tolist()))
            out[f'recall_at_{k}'][row] = 100. * hits / len(truth)
            out[f'precision_at_{k}'][row] = 100. * hits / k
            out[f'f1_at_{k}'][row] = 100. * 2 * hits / (k + len(truth))
            out[f'hit_at_{k}'][row] = 100. * (hits > 0)
    return out


def weighted_auroc(y, p, w):
    """Observation-weighted AUROC with half credit for tied scores."""
    y, p, w = (np.asarray(a, dtype=np.float64) for a in (y, p, w))
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    if not ((y == 1).any() and (y == 0).any()):
        return float('nan')
    order = np.argsort(p, kind='mergesort')
    p, y, w = p[order], y[order], w[order]
    total_neg = float((w * (1 - y)).sum())
    auc, below_neg, i = 0., 0., 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i]:
            j += 1
        tie_neg = float((w[i:j] * (1 - y[i:j])).sum())
        tie_pos_w = w[i:j] * y[i:j]
        auc += float((tie_pos_w * (below_neg + 0.5 * tie_neg)).sum())
        below_neg += tie_neg
        i = j
    total_pos = float((w * y).sum())
    return auc / (total_pos * total_neg)


def weighted_brier(y, p, w):
    """Weighted mean squared error normalised by the weight sum (manuscript Section 2.2)."""
    y, p, w = (np.asarray(a, dtype=np.float64) for a in (y, p, w))
    keep = w > 0
    return float((w[keep] * (p[keep] - y[keep]) ** 2).sum() / w[keep].sum())


def paired_bootstrap(statistic, arrays, n_resamples, seed, alpha=0.05):
    """Percentile interval for ``statistic(*arrays)`` under patient resampling.

    Every array is indexed along axis 0 by the same patient draw, so paired
    comparisons stay paired. Weights are held fixed; the observation model is
    not refitted inside the resample.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(statistic(*arrays))
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        rows = rng.integers(0, n, n)
        draws[b] = statistic(*[a[rows] for a in arrays])
    low, high = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=point, ci_low=float(low), ci_high=float(high), n_patients=int(n),
                n_resamples=int(n_resamples), seed=int(seed))


def mean_difference(a, b):
    return float(np.mean(a) - np.mean(b))
