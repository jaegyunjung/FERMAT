"""
Ablation experiment configs for FERMAT.

These configs define the 6-step ablation that forms the central result:
  1. DX only          — Delphi-equivalent baseline
  2. DX + RX          — adds drug prescriptions
  3. DX + RX + PX     — adds procedures
  4. DX + RX + PX + LAB (static)  — adds screening as one-time baseline
  5. DX + RX + PX + LAB (dynamic) — same biomarkers, updated biennially
  6. Full model       — all modalities including lifestyle

Each config is a dict that can be used to override the base config.
Usage:
  python train.py config/train_fermat_kr.py --ablation=dx_only
"""
from model import TokenType

# Base shared settings
_base = dict(
    dataset='fermat_kr',
    batch_size=128,
    block_size=96,
    n_layer=12,
    n_head=12,
    n_embd=120,
    max_iters=100000,
    mask_ties=True,
)

ABLATION_CONFIGS = {
    # 1. DX only — Delphi equivalent
    'dx_only': {
        **_base,
        'out_dir': 'ablation/dx_only',
        'wandb_run_name': 'ablation-dx-only',
        # Only include DX tokens; everything else gets type-filtered
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
            TokenType.RX, TokenType.PX, TokenType.LAB, TokenType.LIFESTYLE,
        ],
        'vocab_size': 1300,  # ~1200 DX + PAD + NO_EVENT + SEX + DTH
    },

    # 2. DX + RX
    'dx_rx': {
        **_base,
        'out_dir': 'ablation/dx_rx',
        'wandb_run_name': 'ablation-dx-rx',
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
            TokenType.PX, TokenType.LAB, TokenType.LIFESTYLE,
        ],
        'vocab_size': 2800,  # ~1200 DX + ~1500 RX + overhead
    },

    # 3. DX + RX + PX
    'dx_rx_px': {
        **_base,
        'out_dir': 'ablation/dx_rx_px',
        'wandb_run_name': 'ablation-dx-rx-px',
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
            TokenType.LAB, TokenType.LIFESTYLE,
        ],
        'vocab_size': 3300,
    },

    # 4. DX + RX + PX + LAB (static) — LAB tokens only at first screening
    'dx_rx_px_lab_static': {
        **_base,
        'out_dir': 'ablation/dx_rx_px_lab_static',
        'wandb_run_name': 'ablation-static-lab',
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
            TokenType.LIFESTYLE,
        ],
        'vocab_size': 3400,
        # Note: static LAB is achieved in preprocessing, not here.
        # The preprocessor keeps only the first screening LAB tokens per patient.
    },

    # 5. DX + RX + PX + LAB (dynamic) — LAB tokens updated at each screening
    'dx_rx_px_lab_dynamic': {
        **_base,
        'out_dir': 'ablation/dx_rx_px_lab_dynamic',
        'wandb_run_name': 'ablation-dynamic-lab',
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
            TokenType.LIFESTYLE,
        ],
        'vocab_size': 3400,
        # Dynamic LAB: all screening LAB tokens retained (default preprocessing).
    },

    # 6. Full model — all modalities
    'full': {
        **_base,
        'out_dir': 'ablation/full',
        'wandb_run_name': 'ablation-full',
        'ignore_types': [
            TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT,
        ],
        'vocab_size': 3400,
    },
}
