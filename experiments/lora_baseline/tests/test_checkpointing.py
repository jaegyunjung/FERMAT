import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpointing as cp


class CheckpointTests(unittest.TestCase):
    def test_model_configuration_is_bound_independently_of_weights(self):
        import argparse
        import common
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            config = root / 'config/model_config.json'
            config.write_text('{"dropout": 0.1}')
            args = common.add_common_arguments(argparse.ArgumentParser()).parse_args([
                '--data', tmp, '--out', tmp, '--arm', 'lora', '--release', tmp, '--device', 'cpu'])
            args.gpu_profile, args.validation_only = None, True
            first = cp.training_contract(args, {}, {'model_sha256': 'same-weights'})
            config.write_text('{"dropout": 0.2}')
            second = cp.training_contract(args, {}, {'model_sha256': 'same-weights'})
            self.assertEqual(first['identity'], second['identity'])
            self.assertNotEqual(first['model_config_sha256'], second['model_config_sha256'])

    def payload(self, epoch=1):
        return dict(completed_epoch=epoch, contract={'seed': 1}, best_state={'head': torch.tensor([3.])})

    def test_corruption_and_contract_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp.save_epoch(tmp, self.payload(), True)
            manifest = Path(tmp) / 'last_training.json'
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                cp.load_epoch(manifest, {'seed': 2})
            path = Path(tmp) / json.loads(manifest.read_text())['checkpoint']
            with path.open('r+b') as stream:
                stream.seek(100)
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cp.load_epoch(manifest, {'seed': 1})

    def test_failed_publish_keeps_previous_committed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp.save_epoch(tmp, self.payload(), True)
            original = cp.atomic_write
            def fail_manifest(path, writer):
                if Path(path).name == 'last_training.json':
                    raise OSError('simulated power loss before manifest commit')
                return original(path, writer)
            with patch.object(cp, 'atomic_write', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    cp.save_epoch(tmp, self.payload(2), True)
            loaded, _ = cp.load_epoch(Path(tmp) / 'last_training.json', {'seed': 1})
            self.assertEqual(loaded['completed_epoch'], 1)

    def test_atomic_save_does_not_replace_valid_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            cp.atomic_torch_save({'value': 1}, path)
            with patch.object(cp.torch, 'save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    cp.atomic_torch_save({'value': 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)['value'], 1)

    def test_rng_serialization_roundtrip(self):
        rng = np.random.default_rng(123)
        state = cp.capture_rng(rng)
        expected = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rng.pt'
            cp.atomic_torch_save(state, path)
            cp.restore_rng(torch.load(path, weights_only=True), rng)
        actual = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
        self.assertEqual(expected[:3], actual[:3])
        self.assertTrue(torch.equal(expected[3], actual[3]))
