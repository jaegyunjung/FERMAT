import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
