"""Target-free, joint event/time inference for the frozen two-stage FERMAT.

The checkpoint and its training forward remain unchanged. Historical next dates
are observed inputs; the unknown last next-date is marginalized over two branches.
Generation must sample an event conditional on the SAME branch used for time.
"""
from dataclasses import dataclass
import torch
from .model import build_attention_mask

VERSION = 'observed-history-two-date-branches-v1'


@dataclass
class NextDistribution:
    same_day_probability: torch.Tensor
    same_day_events: torch.Tensor
    different_day_events: torch.Tensor
    positive_gap_scale: torch.Tensor
    t_min: float = 0.0

    @property
    def event_probability(self):
        p = self.same_day_probability[:, None]
        return p * self.same_day_events + (1 - p) * self.different_day_events


def _prepare(idx, age, types, block_size):
    if idx.ndim != 2 or idx.shape != age.shape or idx.shape != types.shape:
        raise ValueError('Expected equally shaped [batch, sequence] inputs')
    valid = idx > 0
    if (idx < 0).any():
        raise ValueError('Negative token IDs are invalid')
    if idx.shape[1] == 0 or not valid.any(1).all():
        raise ValueError('Every prefix needs at least one observed token')
    if ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('Only contiguous prefixes with right padding are supported')
    if not torch.isfinite(age[valid]).all():
        raise ValueError('Observed ages must be finite')
    if ((age[:, 1:] < age[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('Observed ages must be ordered')
    lengths = valid.sum(1)
    kept = lengths.clamp_max(block_size)
    n = int(kept.max())
    ix = idx.new_zeros((len(idx), n))
    ag = age.new_full((len(idx), n), -10000.)
    ty = types.new_zeros((len(idx), n))
    for j, length in enumerate(lengths.tolist()):
        start = max(0, length - block_size)
        count = length - start
        ix[j, :count], ag[j, :count], ty[j, :count] = (
            idx[j, start:length], age[j, start:length], types[j, start:length])
    return ix, ag, ty, kept - 1


def _hidden(model, ix, age, types, target_age=None):
    x = model.transformer.token_drop(model.transformer.wte(ix)) * (1 - model.config.token_dropout)
    x = model.transformer.drop(x + model.transformer.wae(age.unsqueeze(-1)) + model.transformer.wtype(types))
    mask = build_attention_mask(ix, age, targets_age=target_age,
                                mask_ties=target_age is not None)
    for block in model.transformer.h:
        x, _ = block(x, mask)
    return model.transformer.ln_f(x)


@torch.inference_mode()
def next_distribution(model, idx, age, types, allowed_token_mask):
    """No targets, target dates, or future outcomes are accepted by this API."""
    if model.training:
        raise ValueError('Inference requires model.eval()')
    if not all(getattr(model.config, name, False) for name in
               ('mask_ties', 'two_stage_time_head', 'decoupled_time_head')):
        raise ValueError('Expected the diagnosed tied-day, two-stage checkpoint')
    ix, ag, ty, last = _prepare(idx, age, types, int(model.config.block_size))
    batch = torch.arange(len(ix), device=ix.device)
    same_dates = ag.clone()
    same_dates[:, :-1] = ag[:, 1:]
    same_dates[batch, last] = ag[batch, last]
    different_dates = same_dates.clone()
    # Only inequality matters to the attention mask; no future date is supplied.
    different_dates[batch, last] = ag[batch, last] + 1.
    hp = _hidden(model, ix, ag, ty)[batch, last]
    hs = _hidden(model, ix, ag, ty, same_dates)[batch, last]
    hd = _hidden(model, ix, ag, ty, different_dates)[batch, last]
    allowed = allowed_token_mask.to(device=ix.device, dtype=torch.bool).clone()
    if allowed.ndim != 1 or len(allowed) != model.config.vocab_size:
        raise ValueError('Vocabulary mask has the wrong shape')
    allowed[sorted(set(model.config.ignore_tokens + model.config.output_ignore_tokens))] = False
    if not allowed.any():
        raise ValueError('No allowed output tokens')
    def probability(h):
        logits = model.lm_head(h).float().masked_fill(~allowed, -torch.inf)
        return logits.softmax(-1)
    p = torch.sigmoid(model.same_day_head(hp).float().squeeze(-1))
    scale = torch.exp(-model.time_head(hd).float().squeeze(-1)) + model.config.t_min
    result = NextDistribution(p, probability(hs), probability(hd), scale,
                              float(model.config.t_min))
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid positive-gap scale')
    return result


def draw_next(distribution, uniforms=None):
    """Draw joint branch, event and gap. Explicit uniforms support exact tests.

    With no uniforms, retain D3's two rand_like + one multinomial call contract
    so both frozen seed-stream schemes remain usable.
    """
    d = distribution
    if uniforms is None:
        same_u = torch.rand_like(d.same_day_probability)
        wait_u = torch.rand_like(d.positive_gap_scale)
    else:
        if uniforms.shape != (len(d.same_day_probability), 3):
            raise ValueError('Expected [batch, 3] uniforms')
        if not torch.isfinite(uniforms).all() or ((uniforms < 0) | (uniforms >= 1)).any():
            raise ValueError('Uniforms must be in [0,1)')
        same_u, wait_u = uniforms[:, 0], uniforms[:, 1]
    same = same_u < d.same_day_probability
    conditional = torch.where(same[:, None], d.same_day_events, d.different_day_events)
    if uniforms is None:
        token = torch.multinomial(conditional, 1).squeeze(-1)
    else:
        cdf = conditional.cumsum(-1)
        cdf[:, -1] = 1.
        token = (cdf <= uniforms[:, 2:3]).sum(-1).clamp_max(cdf.shape[1] - 1)
    wait = -torch.log(wait_u.clamp_min(torch.finfo(wait_u.dtype).tiny)) * d.positive_gap_scale - d.t_min
    gap = torch.where(same, torch.zeros_like(wait), wait.clamp(min=1., max=365*80))
    return token, gap, same


@torch.inference_mode()
def generate_consistent(model, idx, age, token_type, max_new_tokens=100,
                        max_age=85*365.25, no_repeat=False,
                        termination_tokens=None, token_type_lookup=None, top_k=None,
                        temperature=1., allowed_token_mask=None,
                        same_day_no_repeat=False, same_day_repeat_penalty=0.,
                        same_day_temperature=1., same_day_prob_cap=1.,
                        return_final_logits=False, *, distribution_fn=None):
    """Opt-in generation adapter with the validated full-support policy.

    Raw same-day events remain at offset zero, as in the old generator.
    Apply a separately fingerprinted analysis-day convention when evaluating.
    """
    if no_repeat or same_day_no_repeat or same_day_repeat_penalty != 0 or top_k not in (None, 0) or temperature != 1 or same_day_temperature != 1 or same_day_prob_cap != 1:
        raise ValueError('This version implements only the validated full-support policy')
    if return_final_logits:
        raise ValueError('Use next_distribution for target-free probabilities')
    if allowed_token_mask is None or token_type_lookup is None:
        raise ValueError('Explicit clinical vocabulary mask and type lookup required')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    # Legacy D3 passes same-length, unpadded patient prefixes to this entry point.
    if not (idx > 0).all():
        raise ValueError('Generation entry point requires unpadded initial prefixes')
    _prepare(idx, age, token_type, int(model.config.block_size))
    ix, ag, ty = idx.clone(), age.clone(), token_type.clone()
    terminal = torch.as_tensor([] if termination_tokens is None else termination_tokens,
                               device=ix.device, dtype=ix.dtype)
    active = (ag[:, -1] <= max_age) & ~torch.isin(ix, terminal).any(1) & ~(ty == 6).any(1)
    lut = torch.full((model.config.vocab_size,), -1, device=ix.device, dtype=ty.dtype)
    for token, typ in token_type_lookup.items():
        if 0 <= int(token) < len(lut):
            lut[int(token)] = int(typ)
    if (lut[allowed_token_mask.to(device=ix.device, dtype=torch.bool)] < 0).any():
        raise ValueError('Allowed output has no token type')
    for _ in range(max_new_tokens):
        if not active.any():
            break
        predictor = next_distribution if distribution_fn is None else distribution_fn
        d = predictor(model, ix, ag, ty, allowed_token_mask)
        token, gap, _ = draw_next(d)
        last = (ix > 0).sum(1) - 1
        next_age = ag[torch.arange(len(ix), device=ix.device), last] + gap
        keep = active & (next_age <= max_age)
        next_type = lut[token]
        ix = torch.cat([ix, torch.where(keep, token, 0)[:, None]], 1)
        ag = torch.cat([ag, torch.where(keep, next_age, -10000.)[:, None]], 1)
        ty = torch.cat([ty, torch.where(keep, next_type, 0)[:, None]], 1)
        active = keep & ~torch.isin(token, terminal) & (next_type != 6)
    return ix, ag, ty, None
