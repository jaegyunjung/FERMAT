"""
FERMAT: Foundation model for Exploring Real-world Multimodal health data
        using Autoregressive Trajectory modeling.

The causal transformer consumes token, event-type, and continuous-age
representations for diagnosis, prescription, procedure, laboratory, lifestyle,
and death events. Separate prediction heads support clinical-event and event-time
modeling.

The training and evaluation paths support type-specific target policies,
same-day-aware attention, and four-column patient event data:
(patient_id, age_in_days, token_id, token_type_id).
"""

import math
import inspect
from dataclasses import dataclass, field
from enum import IntEnum

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings


# =============================================================================
# Token type definitions
# =============================================================================

class TokenType(IntEnum):
    PAD = 0
    DX = 1        # Diagnosis (KCD / ICD-10)
    RX = 2        # Drug prescription (ATC)
    PX = 3        # Procedure / surgery (EDI)
    LAB = 4       # Lab / screening result (discretized)
    LIFESTYLE = 5 # Lifestyle from screening questionnaire
    DTH = 6       # Death (with cause code)
    SEX = 7       # Sex (static, not predicted)
    NO_EVENT = 8  # No-event padding token (Delphi-style)
    GENOMICS = 9   # Genomics / tumor biomarker token

N_TOKEN_TYPES = len(TokenType)


def build_attention_mask(idx, age, targets_age=None, mask_ties=False):
    """Build the causal mask, optionally hiding events at the target date."""
    batch_size, sequence_length = idx.shape
    causal = torch.tril(
        torch.ones(sequence_length, sequence_length, device=idx.device, dtype=torch.bool)
    )[None, None, :, :]
    valid = idx > 0
    attention_mask = (
        valid.view(batch_size, 1, 1, sequence_length)
        & valid.view(batch_size, 1, sequence_length, 1)
        & causal
    )

    if mask_ties:
        if targets_age is None:
            raise ValueError("targets_age is required when mask_ties=True")
        attention_mask &= (
            age.view(batch_size, 1, 1, sequence_length)
            != targets_age.view(batch_size, 1, sequence_length, 1)
        )
        empty_rows = attention_mask.sum(-1, keepdim=True) == 0
        attention_mask |= empty_rows & torch.eye(
            sequence_length, device=idx.device, dtype=torch.bool
        )[None, None, :, :]

    pad_diagonal = (
        (idx == 0).view(batch_size, 1, 1, sequence_length)
        & torch.eye(sequence_length, device=idx.device, dtype=torch.bool)[None, None, :, :]
    )
    return (attention_mask | pad_diagonal) & causal


def align_time_deltas(age, targets_age, attention_mask, mask_ties):
    """Return waiting times aligned to the latest visible non-tied event."""
    dt = torch.clamp(targets_age - age, min=1.0)
    if not mask_ties or age.shape[1] == 0:
        return dt

    sequence_length = age.shape[1]
    visible_index = (
        attention_mask
        * torch.arange(
            sequence_length,
            device=age.device,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)
    ).max(-1).indices.squeeze(1)
    return torch.gather(dt, -1, visible_index)


def build_target_mask(targets, target_token_type, ignore_tokens, ignore_types):
    """Select target positions that contribute to token and time losses."""
    mask = targets != -1
    for token_id in ignore_tokens:
        mask &= targets != int(token_id)
    if target_token_type is not None:
        for token_type in ignore_types:
            mask &= target_token_type != int(token_type)
    return mask


def safe_masked_mean(values, mask):
    """Return a differentiable zero when a batch has no selected targets."""
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


# =============================================================================
# Modules
# =============================================================================

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    def forward(self, x, attn_mask):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att

class AgeEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term)
        y[..., 1::2] = torch.cos(x / 365.25 * self.div_term)
        y = self.linear(y)
        return y


# =============================================================================
# FERMAT Config & Model
# =============================================================================

@dataclass
class FermatConfig:
    block_size: int = 1024
    vocab_size: int = 3400
    n_token_types: int = N_TOKEN_TYPES
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    bias: bool = True
    t_min: float = 1.0
    log_rate_init: float = None
    decoupled_time_head: bool = False
    two_stage_time_head: bool = False
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    output_ignore_tokens: list = field(default_factory=list)
    ignore_types: list = field(default_factory=lambda: [
        TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT
    ])


class Fermat(nn.Module):
    """
    Causal transformer for longitudinal multimodal clinical event trajectories.

    Each clinical event is represented as the sum of three embeddings:
      - token embedding: the clinical concept or discretized measurement,
      - age encoding: the event time on the patient timeline, and
      - type embedding: DX, RX, PX, LAB, LIFESTYLE, DTH, or metadata.

    The shared representation feeds a next-token head and, when configured,
    separate same-day classification and conditional waiting-time heads.

    Input: idx (B,T), age (B,T), token_type (B,T)
    Output: token logits (B,T,V), multi-objective loss dict, attention weights
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wae = AgeEncoding(config),
            wtype = nn.Embedding(config.n_token_types, config.n_embd),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # Learnable global log event-rate. The per-token logits set the relative
        # ranking of events (cross-entropy); this scalar sets the absolute event
        # rate used by the waiting-time loss. Without it the rate is forced to
        # equal sum(exp(logit)) ~ vocab_size, which is several orders of
        # magnitude above the true rate and makes the time loss dominate and
        # flatten all logits. Initialised to -log(vocab_size) so the initial
        # rate is decoupled from vocabulary size.
        log_rate_init = (
            -math.log(config.vocab_size)
            if config.log_rate_init is None
            else config.log_rate_init
        )
        if config.decoupled_time_head:
            # Option A: a dedicated head predicts the log event-rate from the
            # shared hidden state, so the time loss no longer reshapes the token
            # logits at the output layer. The transformer body stays shared.
            self.time_head = nn.Linear(config.n_embd, 1, bias=True)
        else:
            # Option B (coupled): the rate is the summed token mass shifted by a
            # learnable global scalar.
            self.log_rate = nn.Parameter(torch.tensor(log_rate_init, dtype=torch.float32))
        if config.two_stage_time_head:
            self.same_day_head = nn.Linear(config.n_embd, 1, bias=True)
        self.apply(self._init_weights)
        if config.decoupled_time_head:
            # Start the predicted rate at the same global level as the coupled
            # init (constant in the hidden state) so the time loss is the same
            # order of magnitude as cross-entropy from the first step.
            torch.nn.init.zeros_(self.time_head.weight)
            torch.nn.init.constant_(self.time_head.bias, log_rate_init)
        if config.two_stage_time_head:
            torch.nn.init.zeros_(self.same_day_head.weight)
            torch.nn.init.zeros_(self.same_day_head.bias)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("FERMAT parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, token_type, targets=None, targets_age=None,
                target_token_type=None, validation_loss_mode=False,
                return_attention=True, compute_time_loss=True):
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        type_emb = self.transformer.wtype(token_type)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb + type_emb
        x = self.transformer.drop(x)
        x_input = x

        attn_mask = build_attention_mask(
            idx,
            age,
            targets_age=targets_age,
            mask_ties=targets is not None and self.config.mask_ties,
        )

        attention = [] if return_attention else None
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            if return_attention:
                attention.append(a)
        x = self.transformer.ln_f(x)
        if return_attention:
            attention = torch.stack(attention)

        if targets is not None:
            logits = self.lm_head(x)
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
            output_ignore_tokens = sorted(set(
                self.config.output_ignore_tokens
                + ([1] if validation_loss_mode else [])
            ))
            if output_ignore_tokens:
                logits[..., output_ignore_tokens] = -torch.inf
            targets_flat = targets.reshape(-1)
            target_types_flat = (
                target_token_type.reshape(-1)
                if target_token_type is not None
                else None
            )
            pass_tokens = build_target_mask(
                targets_flat,
                target_types_flat,
                ignored_tokens,
                self.config.ignore_types,
            )
            if pass_tokens.any():
                loss_ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1))[pass_tokens],
                    targets_flat[pass_tokens],
                )
            else:
                loss_ce = x.sum() * 0.0

            if compute_time_loss:
                if self.config.decoupled_time_head:
                    lse = self.time_head(x).squeeze(-1)
                else:
                    lse = torch.logsumexp(logits, -1) + self.log_rate
                lse = -torch.log(torch.exp(-lse) + self.config.t_min)
                dt = align_time_deltas(
                    age,
                    targets_age,
                    attn_mask,
                    self.config.mask_ties,
                )
                ldt = -torch.log(dt + self.config.t_min).view(-1)
                per_target_dt = -(
                    lse.reshape(-1)
                    - torch.exp(lse.reshape(-1) - ldt.reshape(-1))
                )
                if self.config.two_stage_time_head:
                    # The same-day label is exactly what the targets_age tie mask
                    # encodes: it hides current-date events only when the next
                    # event is same-day. Reusing the main hidden state would let
                    # the same-day head read its own answer off the attention
                    # pattern (label leakage). Recompute a target-independent
                    # plain-causal representation for this head only.
                    causal_mask = build_attention_mask(idx, age, mask_ties=False)
                    x_same_day = x_input
                    for block in self.transformer.h:
                        x_same_day, _ = block(x_same_day, causal_mask)
                    x_same_day = self.transformer.ln_f(x_same_day)
                    same_day_logits = self.same_day_head(x_same_day).squeeze(-1)
                    same_day_targets = targets_age == age
                    clinical_time_tokens = torch.zeros_like(pass_tokens)
                    if target_types_flat is not None:
                        for token_type in (
                            TokenType.DX,
                            TokenType.RX,
                            TokenType.PX,
                            TokenType.DTH,
                        ):
                            clinical_time_tokens |= (
                                pass_tokens
                                & (target_types_flat == int(token_type))
                            )
                    else:
                        clinical_time_tokens = pass_tokens
                    per_target_same_day = F.binary_cross_entropy_with_logits(
                        same_day_logits.reshape(-1),
                        same_day_targets.reshape(-1).to(same_day_logits.dtype),
                        reduction="none",
                    )
                    loss_same_day = safe_masked_mean(
                        per_target_same_day,
                        clinical_time_tokens,
                    )
                    different_day_tokens = (
                        clinical_time_tokens & ~same_day_targets.reshape(-1)
                    )
                    loss_dt = safe_masked_mean(
                        per_target_dt,
                        different_day_tokens,
                    )
                else:
                    same_day_logits = None
                    loss_same_day = loss_ce.new_zeros(())
                    clinical_time_tokens = pass_tokens
                    different_day_tokens = pass_tokens
                    loss_dt = safe_masked_mean(per_target_dt, pass_tokens)
                # Expose the per-position effective log-rate (post t_min cap) so
                # evaluation can derive the predicted waiting time without
                # re-deriving the rate, which differs between the coupled and
                # decoupled heads.
                effective_log_rate = lse
            else:
                loss_dt = loss_ce.new_zeros(())
                loss_same_day = loss_ce.new_zeros(())
                effective_log_rate = None
                same_day_logits = None
                clinical_time_tokens = pass_tokens.new_zeros(pass_tokens.shape)
                different_day_tokens = pass_tokens.new_zeros(pass_tokens.shape)
            loss = {
                'loss_ce': loss_ce,
                'loss_same_day': loss_same_day,
                'loss_dt': loss_dt,
                'n_targets': pass_tokens.sum(),
                'n_time_targets': clinical_time_tokens.sum(),
                'n_dt_targets': different_day_tokens.sum(),
                'effective_log_rate': effective_log_rate,
                'same_day_logits': same_day_logits,
            }
        else:
            logits = self.lm_head(x[:, :, :])
            loss = None

        return logits, loss, attention

    def adjust_block_size(self, block_size):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (torch.nn.Linear,)):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)):
                    no_decay.add(fpn)
                elif pn == 'log_rate':
                    no_decay.add(fpn)
        decay.remove('lm_head.weight')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(self, idx, age, token_type, max_new_tokens=100, max_age=85*365.25,
                 no_repeat=True, termination_tokens=None, token_type_lookup=None,
                 top_k=None, temperature=1.0, allowed_token_mask=None,
                 same_day_no_repeat=False, same_day_repeat_penalty=0.0,
                 same_day_temperature=1.0, same_day_prob_cap=1.0):
        if termination_tokens is None:
            warnings.warn('Set termination_tokens for your vocabulary.')
            termination_tokens = []
        if token_type_lookup is None:
            warnings.warn('token_type_lookup not provided. Defaulting generated tokens to DX.')
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000
        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            block_size = int(self.config.block_size)
            idx_in = idx[:, -block_size:]
            age_in = age[:, -block_size:]
            token_type_in = token_type[:, -block_size:]

            tok_emb = self.transformer.wte(idx_in)
            age_emb = self.transformer.wae(age_in.unsqueeze(-1))
            type_emb = self.transformer.wtype(token_type_in)
            x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
            x = x + age_emb + type_emb
            x = self.transformer.drop(x)
            attn_mask = build_attention_mask(idx_in, age_in, mask_ties=False)
            for block in self.transformer.h:
                x, _ = block(x, attn_mask)
            x = self.transformer.ln_f(x)

            logits = self.lm_head(x)[:, -1, :]
            ignored_outputs = sorted(set(
                self.config.ignore_tokens
                + self.config.output_ignore_tokens
            ))
            logits[:, ignored_outputs] = -torch.inf
            if allowed_token_mask is not None:
                logits[:, ~allowed_token_mask.to(device=logits.device, dtype=torch.bool)] = -torch.inf
            if no_repeat:
                fill = idx.clone(); fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            if same_day_no_repeat or same_day_repeat_penalty > 0:
                adjusted = logits.clone()
                for batch_index in range(idx.size(0)):
                    same_day_positions = torch.isclose(
                        age[batch_index],
                        age[batch_index, -1],
                        atol=1e-4,
                        rtol=0.0,
                    )
                    same_day_tokens = idx[batch_index, same_day_positions]
                    same_day_tokens = same_day_tokens[same_day_tokens > 1]
                    if same_day_tokens.numel() == 0:
                        continue
                    if same_day_no_repeat:
                        adjusted[batch_index, same_day_tokens] = -torch.inf
                    else:
                        adjusted[batch_index, same_day_tokens] = (
                            adjusted[batch_index, same_day_tokens]
                            - float(same_day_repeat_penalty)
                        )
                logits = adjusted
            if self.config.decoupled_time_head:
                raw_log_rate = self.time_head(x[:, -1, :]).squeeze(-1)
                log_rate = -torch.log(torch.exp(-raw_log_rate) + self.config.t_min)
                rate = torch.exp(log_rate).clamp_min(1e-12)
                if self.config.two_stage_time_head:
                    same_day_logit = self.same_day_head(x[:, -1, :]).squeeze(-1)
                    same_day_logit = same_day_logit / max(float(same_day_temperature), 1e-6)
                    same_day_prob = torch.sigmoid(same_day_logit)
                    same_day_prob = torch.clamp(same_day_prob, max=float(same_day_prob_cap))
                    same_day = torch.rand_like(same_day_prob) < same_day_prob
                else:
                    same_day = torch.zeros_like(rate, dtype=torch.bool)
                wait = -torch.rand_like(rate).clamp_min(1e-12).log() / rate
                wait = wait - self.config.t_min
                wait = torch.where(
                    same_day,
                    torch.zeros_like(wait),
                    torch.clamp(wait, min=1.0),
                )
                wait = torch.clamp(wait, min=0.0, max=365 * 80)
                sample_logits = logits / max(float(temperature), 1e-6)
                if top_k is not None and top_k > 0 and top_k < sample_logits.size(-1):
                    values, _ = torch.topk(sample_logits, top_k, dim=-1)
                    threshold = values[:, [-1]]
                    sample_logits = sample_logits.masked_fill(sample_logits < threshold, -torch.inf)
                probs = F.softmax(sample_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                age_next = age[..., [-1]] + wait[:, None]
            else:
                rate_logits = logits + self.log_rate
                if top_k is not None and top_k > 0 and top_k < rate_logits.size(-1):
                    values, _ = torch.topk(rate_logits, top_k, dim=-1)
                    threshold = values[:, [-1]]
                    rate_logits = rate_logits.masked_fill(rate_logits < threshold, -torch.inf)
                t_next = torch.clamp(
                    -torch.exp(-rate_logits) * torch.rand(rate_logits.shape, device=idx.device).clamp_min(1e-12).log(),
                    min=0,
                    max=365 * 80,
                ).min(1)
                idx_next = t_next[1][:, None]
                age_next = age[..., [-1]] + t_next[0][:, None]
            if token_type_lookup is not None:
                type_next = torch.tensor([[token_type_lookup.get(int(i), TokenType.DX)] for i in idx_next.squeeze(-1)], device=idx.device, dtype=torch.long)
            else:
                type_next = torch.full_like(idx_next, TokenType.DX)
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            token_type = torch.cat((token_type, type_next), dim=1)
            if torch.logical_or(
                type_next.squeeze(-1) == int(TokenType.DTH),
                age_next.squeeze(-1) > max_age,
            ).all():
                break
            if len(termination_tokens) > 0 and torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break

        if len(termination_tokens) > 0:
            pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)
        else:
            pad = age > max_age
        logits, _, _ = self(
            idx,
            age,
            token_type,
            return_attention=False,
        )
        idx[pad] = 0; age[pad] = mask_time; token_type[pad] = TokenType.PAD
        if no_repeat:
            fill = idx + 0; fill[fill == 1] = 0
            logits = torch.stack([logits[:, j].scatter_(1, fill[:, :j+1], -torch.inf) for j in range(fill.shape[1])]).transpose(0, 1)
        return idx, age, token_type, logits
