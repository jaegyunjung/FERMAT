"""Low-rank adaptation (LoRA; Hu et al., 2021) for the frozen FERMAT release.

Base weights are never modified. Each selected ``nn.Linear`` is wrapped so that
``y = W x + (alpha / r) * B A x`` with ``B`` initialised to zero, so an injected
model reproduces the released outputs exactly until the first optimizer step.
The wrapper keeps the original module as ``.base`` and its parameters frozen;
``model.state_dict()`` contains the unchanged release tensors under ``.base``
names plus ``lora_A`` / ``lora_B`` for the adapters.

The released FERMAT attention block fuses query, key and value into one
``c_attn`` projection (640 -> 1920). A LoRA on ``c_attn`` therefore adapts
q, k and v jointly with one shared rank-r subspace; ``c_proj`` is the output
projection. ``attention`` (the protocol's primary arm) adapts both;
``attention_mlp`` additionally adapts the feed-forward projections.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

TARGET_SETS = {
    'attention': ('attn.c_attn', 'attn.c_proj'),
    'attention_mlp': ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'),
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRA wraps nn.Linear modules only')
        if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('rank must be a positive integer and alpha positive and finite')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        # Kaiming-uniform A and zero B is the reference initialisation: the
        # product starts at zero. Initially only B receives a nonzero gradient.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merged_weight(self):
        """Dense ``W + scaling * B A`` for export; the stored base stays frozen."""
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


def _resolve(model, name):
    parent = model
    parts = name.split('.')
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model, rank=8, alpha=16., dropout=0.05, targets='attention'):
    """Freeze every base parameter, then wrap the selected projections in place.

    Returns the list of wrapped module names. Raises if the model already holds
    adapters, so a second call cannot stack two LoRAs silently.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('Model already contains LoRA adapters')
    suffixes = TARGET_SETS[targets] if isinstance(targets, str) else tuple(targets)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.startswith('transformer.h.') \
                and name.endswith(suffixes):
            parent, attribute = _resolve(model, name)
            adapter = LoRALinear(module, rank, alpha, dropout)
            adapter.to(module.weight.device)
            setattr(parent, attribute, adapter)
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f'No modules matched LoRA targets {suffixes}')
    return wrapped


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n]


def lora_state_dict(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if 'lora_A' in n or 'lora_B' in n}


def load_lora_state_dict(model, state):
    current = {n: p for n, p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n}
    if set(state) != set(current):
        raise ValueError('LoRA state does not match the injected adapters')
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(current[name].shape):
            raise ValueError(f'LoRA rank mismatch at {name}: {tuple(tensor.shape)} vs {tuple(current[name].shape)}')
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name].to(parameter.device))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, trainable_fraction=trainable / total)


def snapshot_parameters(model):
    """Copy of every parameter, taken before ``inject_lora`` for the final audit."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def assert_base_unchanged(model, snapshot):
    """Raise unless every pre-injection parameter is bit-identical after training."""
    current = {n: p for n, p in model.named_parameters()}
    for name, tensor in snapshot.items():
        wrapped_name = name
        # c_attn.weight becomes c_attn.base.weight after injection.
        if name not in current:
            head, tail = name.rsplit('.', 1)
            wrapped_name = f'{head}.base.{tail}'
        if wrapped_name not in current:
            raise AssertionError(f'Parameter missing after injection: {name}')
        if not torch.equal(current[wrapped_name].detach().cpu(), tensor):
            raise AssertionError(f'Frozen parameter changed: {name}')
