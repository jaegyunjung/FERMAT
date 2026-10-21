"""Load the standalone, tensor-only release; no access to the training database."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .model import Fermat, FermatConfig
from .inference import _prepare, _hidden


def verify_artifacts(root):
    root = Path(root)
    manifest = json.loads((root / 'config/artifact_manifest.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def load_release(root, device='cpu', verify=True):
    root = Path(root)
    if verify:
        verify_artifacts(root)
    config = FermatConfig(**json.loads((root / 'config/model_config.json').read_text()))
    model = Fermat(config)
    state = torch.load(root / 'artifacts/model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    # FP32 parameters match the original evaluation loader. BF16 inference, if
    # desired, should use torch.autocast rather than silently changing weights.
    return model.to(device).eval()


@torch.inference_mode()
def embedding(model, idx, age, token_type):
    if model.training:
        raise ValueError('Call model.eval() before extracting frozen embeddings')
    idx, age, token_type, last = _prepare(idx, age, token_type, model.config.block_size)
    hidden = _hidden(model, idx, age, token_type)
    return hidden[torch.arange(len(idx), device=idx.device), last]


class NextRecordHead(torch.nn.Module):
    def __init__(self, state, mean, sd, classes):
        super().__init__()
        self.linear = torch.nn.Linear(len(mean), len(classes))
        self.linear.load_state_dict(state, strict=True)
        self.register_buffer('mean', torch.as_tensor(mean.copy(), dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd.copy(), dtype=torch.float32))
        self.register_buffer('classes', torch.as_tensor(classes.copy(), dtype=torch.long))

    def forward(self, embeddings):
        return self.linear((embeddings.float() - self.mean) / self.sd)

    def top_tokens(self, embeddings, k=10):
        # Stable class-order tie break for new recipients; the historical
        # evaluation used a patient-seeded tie break (see MODEL_CARD.md).
        order = torch.argsort(self(embeddings), descending=True, stable=True)[:, :k]
        return self.classes[order]


def load_next_record_head(root, device='cpu'):
    root = Path(root)
    verify_artifacts(root)
    with np.load(root / 'artifacts/head_standardization.npz', allow_pickle=False) as norm:
        mean, sd = norm['mean'], norm['sd']
    if mean.shape != (640,) or sd.shape != (640,) or not np.isfinite(mean).all() or not (sd > 0).all():
        raise ValueError('Invalid head normalization')
    classes = np.load(root / 'artifacts/head_classes.npy', allow_pickle=False)
    state = torch.load(root / 'artifacts/next_record_head.pt', map_location='cpu', weights_only=True)
    return NextRecordHead(state, mean, sd, classes).to(device).eval()
