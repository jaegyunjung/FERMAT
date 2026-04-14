"""
FERMAT data utilities.

Extends Delphi's data loading to handle 4-column format:
  (patient_id, age_in_days, token_id, token_type_id)

Also provides backward compatibility with Delphi's 3-column format.
"""

import numpy as np
import torch
import re

from model import TokenType


def load_data(path):
    """
    Load a .bin data file. Auto-detects 3-column (Delphi) vs 4-column (FERMAT) format.

    Returns:
        data: np.ndarray of shape (N, 3) or (N, 4)
        has_types: bool indicating whether token types are present
    """
    raw = np.memmap(path, dtype=np.uint32, mode='r')
    if raw.shape[0] % 4 == 0 and raw.shape[0] % 3 != 0:
        data = raw.reshape(-1, 4)
        has_types = True
    elif raw.shape[0] % 3 == 0 and raw.shape[0] % 4 != 0:
        data = raw.reshape(-1, 3)
        has_types = False
    else:
        # Ambiguous — try 4-column first, check if 4th column looks like type IDs
        data4 = raw.reshape(-1, 4)
        if data4[:, 3].max() < 20:  # type IDs should be small
            data = data4
            has_types = True
        else:
            data = raw.reshape(-1, 3)
            has_types = False

    return data, has_types


def get_p2i(data):
    """
    Get the patient-to-index mapping.
    Returns array of (start_idx, length) pairs.
    """
    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def get_batch(ix, data, p2i, select='left', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False,
              no_event_token_rate=5, cut_batch=False):
    """
    Get a batch of data. Extends Delphi's get_batch to handle 4-column data.

    Returns:
        x: (B, T) input token IDs
        a: (B, T) input ages
        y: (B, T) target token IDs
        b: (B, T) target ages
        xt: (B, T) input token type IDs  [NEW — None if 3-column data]
    """
    has_types = data.shape[1] == 4
    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) %
                                         torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    if has_types:
        types = torch.from_numpy(data[:, 3][batch_idx].astype(np.int64))
    else:
        types = None

    # Augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        if has_types:
            lifestyle_idx = (types == TokenType.LIFESTYLE) | (types == TokenType.SEX)
        else:
            # Delphi convention: tokens 3-11 are lifestyle
            lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)
    if has_types:
        types = types.masked_fill(~mask, TokenType.PAD)

    # Insert "no event" padding tokens
    if (padding.lower() == 'none' or padding is None or
            no_event_token_rate == 0 or no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values

    # Stack no-event tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])
    if has_types:
        types = torch.hstack([types, torch.full_like(pad, TokenType.NO_EVENT, dtype=torch.int)])

    # Mask no-event tokens beyond last real token
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)
    if has_types:
        types = types.masked_fill(ages <= mask_time, TokenType.PAD)

    # Sort by age
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)
    if has_types:
        types = torch.gather(types, 1, s)

    # Token 0 is reserved for padding — shift all tokens by 1
    tokens = tokens + 1

    # Cut padded prefix
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if has_types:
            types = types[:, cut_margin:]

    # Cut to block_size + 1
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if has_types:
            types = types[:, cut_margin:]

    # Shift by one to create (input, target) pairs
    x_tok = tokens[:, :-1]
    a = ages[:, :-1]
    y_tok = tokens[:, 1:]
    b = ages[:, 1:]
    if has_types:
        xt = types[:, :-1]
    else:
        xt = None

    # Mask leading no-event tokens
    x_tok = x_tok.masked_fill((x_tok == 0) * (y_tok == 1), 0)
    y_tok = y_tok.masked_fill(x_tok == 0, 0)
    b = b.masked_fill(x_tok == 0, mask_time)

    if device == 'cuda':
        tensors = [x_tok, a, y_tok, b]
        if xt is not None:
            tensors.append(xt)
        tensors = [t.pin_memory().to(device, non_blocking=True) for t in tensors]
        if xt is not None:
            x_tok, a, y_tok, b, xt = tensors
        else:
            x_tok, a, y_tok, b = tensors
    else:
        x_tok, a, y_tok, b = x_tok.to(device), a.to(device), y_tok.to(device), b.to(device)
        if xt is not None:
            xt = xt.to(device)

    return x_tok, a, y_tok, b, xt


# =============================================================================
# SHAP utilities (kept from Delphi for compatibility)
# =============================================================================

def shap_custom_tokenizer(s, return_offsets_mapping=True):
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {"input_ids": input_ids}
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out
