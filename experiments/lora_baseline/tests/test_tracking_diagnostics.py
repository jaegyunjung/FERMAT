import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import subprocess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import Tracker
from runtime_guard import validate_sample
from train_record import load_diagnostic_embeddings
from gates import make_gate, official_targets, validate_gate
import test_reproduction


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostic_recipe_cannot_issue_real_gate(self):
        log = test_reproduction.GateRegressionTests().log()
        log['fingerprint']['recipe'] = 'diagnostic_v1'
        gate = make_gate(log, official_targets('record', None))
        self.assertFalse(gate['passed'])
        self.assertTrue(validate_gate(gate, log['fingerprint']['gate_identity']))

    def test_tracker_rejects_patient_data_and_retains_local_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Tracker(tmp)
            for metrics in ({'patient_id': 12}, {'test/raw': [1, 2]}, {'train/loss': float('nan')}):
                with self.assertRaises(ValueError):
                    t.log(metrics)
            t.log({'validation/recall_at_10': 55., 'gate/passed': False})
            t.finish()
            saved = json.loads((Path(tmp) / 'aggregate_history.jsonl').read_text())
            self.assertEqual(saved['validation/recall_at_10'], 55.)

    def test_gpu_limits_reject_bad_samples(self):
        validate_sample([400, 100, 45, 1500, 2000])
        for sample in ([400, 331, 45, 1500, 2000], [400, 100, 71, 1500, 2000],
                       [400, 100, 45, 1600, 2000], [400, 100, 45, 1500, 31000]):
            with self.assertRaises(RuntimeError):
                validate_sample(sample)

    def test_cache_requires_exact_patient_order(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'train'
            folder.mkdir()
            np.save(folder / 'patient_id.npy', np.array([2, 1]))
            with self.assertRaisesRegex(ValueError, 'order'):
                load_diagnostic_embeddings(tmp, {'train': SimpleNamespace(patient_id=np.array([1, 2]))},
                                           {'train': np.array([0, 1])}, 3)

    def test_diagnostic_scaler_includes_training_patients_without_targets(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            subprocess.run([sys.executable, str(package / 'make_synthetic_dataset.py'), '--out', str(data),
                            '--patients', '8', '8', '8', '--max-length', '8'], check=True, capture_output=True)
            config_path = package / 'tests/smoke_model_config.json'
            config = json.loads(config_path.read_text())
            width = config['n_embd']
            for split in ('train', 'validation', 'test'):
                np.save(data / split / 'frozen_embeddings.npy', np.zeros((8, width), dtype=np.float32))
            pool = data / 'preprocessing_train'
            pool.mkdir()
            for name in ('idx', 'age', 'token_type', 'patient_id'):
                array = np.load(data / 'train' / (name + '.npy'))
                extra = array[:1].copy()
                if name == 'patient_id':
                    extra[:] = 999999
                np.save(pool / (name + '.npy'), np.concatenate([array, extra]))
            values = np.zeros((9, width), dtype=np.float32)
            values[-1] = 90.
            np.save(pool / 'frozen_embeddings.npy', values)
            out = root / 'run'
            result = subprocess.run([sys.executable, str(package / 'train_record.py'), '--data', str(data),
                '--out', str(out), '--arm', 'head_only', '--recipe', 'diagnostic_v1', '--embeddings-cache', str(data),
                '--device', 'cpu', '--epochs', '1', '--smoke-random-init', str(config_path)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            state = torch.load(out / 'trainable_state.pt', weights_only=True)
            np.testing.assert_array_equal(state['preprocessing']['embedding_mean'], np.full(width, 10., np.float32))
            self.assertFalse((out / 'gate.json').exists())
            from data import PrefixSplit, dataset_content_hash
            splits = {s: PrefixSplit(data, s) for s in ('train', 'validation', 'test', 'preprocessing_train')}
            before = dataset_content_hash(splits, 'record', None)
            age = np.load(pool / 'age.npy')
            age[-1, 0] += 1
            np.save(pool / 'age.npy', age)
            self.assertNotEqual(before, dataset_content_hash(splits, 'record', None))


if __name__ == '__main__':
    unittest.main()
