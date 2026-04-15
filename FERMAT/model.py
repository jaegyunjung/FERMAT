"""
FERMAT: Foundation model for Exploring Real-world Multimodal health data
        using Autoregressive Trajectory modeling.

Extends the Delphi architecture (Shmatko, Jung, Gaurav et al., Nature 2025) with:
  1. Token Type Embedding — distinguishes DX, RX, PX, LAB, LIFESTYLE, DTH, PAD
  2. 4-column data format — (patient_id, age_in_days, token_id, token_type_id)
  3. Type-aware ignore_tokens — per-type control over loss computation

Based on nanoGPT (Karpathy) and Delphi (gerstung-lab).
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

N_TOKEN_TYPES = len(TokenType)


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
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    ignore_types: list = field(default_factory=lambda: [
        TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT
    ])


class Fermat(nn.Module):
    """
    FERMAT: Foundation model for Exploring Real-world Multimodal health data
            using Autoregressive Trajectory modeling.

    Each clinical event is represented as the sum of three embeddings:
      - Token embedding:  what specific clinical code (e.g., E11 = T2DM)
      - Age encoding:     when it occurred (continuous sinusoidal)
      - Type embedding:   what kind of event (DX, RX, PX, LAB, etc.)

    Input:  idx (B,T), age (B,T), token_type (B,T)
    Output: logits (B,T,V), loss dict, attention weights
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
        self.apply(self._init_weights)
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
                validation_loss_mode=False):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        type_emb = self.transformer.wtype(token_type)
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb + type_emb
        x = self.transformer.drop(x)

        attn_mask = ((idx > 0).view(b, 1, 1, t) * (idx > 0).view(b, 1, t, 1))
        attn_mask = attn_mask * (torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0)
        if targets is not None and self.config.mask_ties:
            attn_mask = attn_mask * (age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1))
            attn_mask = attn_mask + (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask + (idx == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask * (torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0)

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            logits = self.lm_head(x)
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[..., ignored_tokens] = -torch.inf
            targets_flat = targets.reshape(-1)
            pass_tokens = targets_flat != -1
            for k in ignored_tokens:
                pass_tokens = pass_tokens & (targets_flat != k)

            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets_flat[pass_tokens], ignore_index=-1)

            lse = torch.logsumexp(logits, -1)
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, t, device=device, dtype=torch.float32).view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2)))
            ldt = -torch.log(dt + self.config.t_min).view(-1)
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1)))
            loss_dt = torch.mean(loss_dt[pass_tokens])
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
        else:
            logits = self.lm_head(x[:, :, :])
            loss = None

        return logits, loss, att

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
                 no_repeat=True, termination_tokens=None, token_type_lookup=None, top_k=None):
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
            logits, _, _ = self(idx, age, token_type)
            logits = logits[:, -1, :]
            logits[:, self.config.ignore_tokens] = -torch.inf
            if no_repeat:
                fill = idx.clone(); fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]
            if token_type_lookup is not None:
                type_next = torch.tensor([[token_type_lookup.get(int(i), TokenType.DX)] for i in idx_next.squeeze(-1)], device=idx.device, dtype=torch.long)
            else:
                type_next = torch.full_like(idx_next, TokenType.DX)
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            token_type = torch.cat((token_type, type_next), dim=1)
            if len(termination_tokens) > 0 and torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break

        if len(termination_tokens) > 0:
            pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)
        else:
            pad = age > max_age
        logits, _, _ = self(idx, age, token_type)
        idx[pad] = 0; age[pad] = mask_time; token_type[pad] = TokenType.PAD
        if no_repeat:
            fill = idx + 0; fill[fill == 1] = 0
            logits = torch.stack([logits[:, j].scatter_(1, fill[:, :j+1], -torch.inf) for j in range(fill.shape[1])]).transpose(0, 1)
        return idx, age, token_type, logits
