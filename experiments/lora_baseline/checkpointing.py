"""Transactional epoch checkpoints. The atomic manifest is the commit point."""
import hashlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(value, path):
    atomic_write(path, lambda stream: torch.save(value, stream))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def capture_rng(generator):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]),
                generator=generator.bit_generator.state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    generator.bit_generator.state = state['generator']
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def training_contract(args, cfg, identity, diagnostic_hashes=None):
    here = Path(__file__).resolve().parent
    names = ('train_record.py', 'common.py', 'checkpointing.py', 'data.py', 'recipes.py', 'lora.py', 'metrics.py')
    if identity.get('task') == 'risk':
        names = ('train_risk.py', 'risk_cache.py', 'common.py', 'checkpointing.py', 'data.py', 'recipes.py', 'lora.py', 'metrics.py')
    code = {name: sha256(here / name) for name in names}
    release_code = here.parents[1] / 'releases/fermat_snuh_20261021/fermat_release'
    code.update({'release/' + f.name: sha256(f) for f in sorted(release_code.glob('*.py'))})
    fields = ('arm', 'recipe', 'seed', 'smoke_seed', 'rank', 'alpha', 'targets', 'lora_dropout',
              'lr_lora', 'batch_size', 'grad_accum', 'warmup_steps', 'grad_clip', 'betas', 'bf16',
              'validation_only')
    if identity.get('task') == 'risk':
        fields += ('disease', 'matched_head', 'risk_steps', 'risk_c_values')
    profile = json.loads(args.gpu_profile.read_text(encoding='utf-8-sig')) if args.gpu_profile else None
    model_config = args.smoke_random_init or (args.release / 'config/model_config.json')
    contract = dict(code=code, identity={k: v for k, v in identity.items() if k != 'dataset_hash_seconds'},
                    model_config_sha256=sha256(model_config),
                    settings={k: getattr(args, k) for k in fields}, resolved=cfg,
                    diagnostic_cache=diagnostic_hashes, python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, cuda=torch.version.cuda, device=str(args.device),
                    num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
                    environment={k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE')},
                    gpu_profile={k: profile[k] for k in ('driver_version', 'power_limit_w', 'max_clock_mhz')} if profile else None,
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None)
    return json.loads(json.dumps(contract))


def save_epoch(out, payload, improved):
    out = Path(out)
    relative = f'checkpoints/epoch_{payload["completed_epoch"]:04d}.pt'
    target = out / relative
    if target.exists():
        raise ValueError('Refusing to overwrite immutable epoch checkpoint')
    payload = dict(payload, format_version=FORMAT_VERSION)
    atomic_torch_save(payload, target)
    # Read back before publishing a manifest, so incomplete files are never resumable.
    saved = torch.load(target, map_location='cpu', weights_only=True)
    if saved['completed_epoch'] != payload['completed_epoch'] or saved['contract'] != payload['contract']:
        raise RuntimeError('Checkpoint readback failed')
    if improved:
        atomic_torch_save(payload['best_state'], out / 'best_model.pt')
    manifest = dict(format_version=FORMAT_VERSION, checkpoint=relative,
                    sha256=sha256(target), completed_epoch=payload['completed_epoch'])
    atomic_write(out / 'last_training.json', lambda stream: stream.write(json.dumps(manifest, indent=2).encode()))
    return manifest


def load_epoch(manifest_path, expected_contract):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unknown resume manifest format')
    path = (manifest_path.parent / manifest['checkpoint']).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()) or sha256(path) != manifest['sha256']:
        raise ValueError('Resume checkpoint path or checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != FORMAT_VERSION or state['completed_epoch'] != manifest['completed_epoch']:
        raise ValueError('Resume checkpoint metadata mismatch')
    if state['contract'] != expected_contract:
        changed = sorted(k for k in set(state['contract']) | set(expected_contract)
                         if state['contract'].get(k) != expected_contract.get(k))
        raise ValueError('Resume contract mismatch: ' + ', '.join(changed))
    return state, manifest
