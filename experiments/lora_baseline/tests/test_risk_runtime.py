import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from audit_record_resume import tree_equal
from risk_cache import source_identity
from prepare_record_v1 import sha256
from recipes import fit_risk_scaler
import numpy as np


class RiskRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = cls.root / 'data'
        cls.call('make_synthetic_dataset.py', '--out', cls.data, '--patients', '40', '16', '16', '--max-length', '24')
        cls.head = cls.root / 'gate_head'
        cls.call('train_risk.py', *cls.args(cls.head, 'head_only'))
        summary = json.loads((cls.head / 'summary.json').read_text())['summary']
        cls.gate = cls.root / 'gate.json'
        extra = []
        for key in ('weighted_auroc', 'weighted_brier'):
            extra += ['--metric', 'test.'+key, '--expected', str(summary['test'][key]), '--tolerance', '1e-9']
        cls.call('check_gate.py', '--summary', cls.head / 'summary.json', '--out', cls.gate, *extra)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def call(cls, script, *args, fails=False):
        p = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True)
        if (p.returncode != 0) != fails:
            raise AssertionError(p.stdout+p.stderr)
        return p.stdout+p.stderr

    @classmethod
    def args(cls, out, arm='lora'):
        return ['--out', out, '--data', cls.data, '--arm', arm, '--disease', 'ckd', '--device', 'cpu',
                '--smoke-random-init', HERE / 'tests/smoke_model_config.json', '--batch-size', '8',
                '--epochs', '2', '--patience', '3', '--seed', '1']

    def test_resume_and_independent_reload(self):
        def args(out):
            return self.args(out)+['--gate', self.gate, '--validation-only', '--grad-accum', '3']
        full, paused, resumed = [self.root / s for s in ('full', 'paused', 'resumed')]
        self.call('train_risk.py', *args(full))
        self.call('train_risk.py', *args(paused), '--stop-after-epoch', '1')
        self.call('train_risk.py', *args(resumed), '--resume', paused / 'last_training.json')
        def latest(p):
            m = json.loads((p/'last_training.json').read_text())
            return torch.load(p/m['checkpoint'], weights_only=True)
        a, b = latest(full), latest(resumed)
        for key in ('current_state', 'optimizer', 'best_state', 'best', 'since_best', 'optimizer_steps', 'rng', 'contract'):
            self.assertTrue(tree_equal(a[key], b[key]), key)
        self.assertIn('train_risk.py', b['contract']['code'])
        self.assertIn('risk_cache.py', b['contract']['code'])
        self.assertTrue(tree_equal(torch.load(full/'validation_scores.pt', weights_only=True),
                                   torch.load(resumed/'validation_scores.pt', weights_only=True)))
        self.call('predict.py', '--state', resumed/'trainable_state.pt', '--data', self.data,
                  '--out', self.root/'reload', '--device', 'cpu', '--split', 'validation', '--batch-size', '8',
                  '--reference-scores', resumed/'validation_scores.pt',
                  '--smoke-random-init', HERE/'tests/smoke_model_config.json')
        self.assertTrue(json.loads((self.root/'reload/reload_audit.json').read_text())['passed'])
        for folder in (full, paused, resumed):
            for name in ('summary.json', 'test_scores.pt', 'test_patient_probabilities.csv', 'gate.json'):
                self.assertFalse((folder/name).exists())
        error = self.call('train_risk.py', *args(self.root/'wrong'), '--resume', paused/'last_training.json', '--seed', '2', fails=True)
        self.assertIn('contract mismatch', error)
        error = self.call('train_risk.py', *self.args(self.root/'not_validation'), '--gate', self.gate,
                          '--grad-accum', '3', '--resume', paused/'last_training.json', fails=True)
        self.assertIn('contract mismatch', error)
        self.call('train_risk.py', *args(self.root/'terminal'), '--resume', resumed/'last_training.json')
        self.assertEqual(len(json.loads((self.root/'terminal/validation_summary.json').read_text())['epochs']), 2)

    def test_matched_control_and_benchmark(self):
        out = self.root/'matched'
        self.call('train_risk.py', *self.args(out, 'head_only'), '--recipe', 'experimental', '--matched-head', '--validation-only')
        summary = json.loads((out/'validation_summary.json').read_text())
        self.assertEqual(summary['fingerprint']['warmup_steps'], 100)
        self.assertEqual(summary['fingerprint']['resolved']['lr_head'], .001)
        self.assertEqual(summary['fingerprint']['grad_clip'], 1.)
        self.assertFalse((out/'gate.json').exists())
        out = self.root/'benchmark'
        self.call('train_risk.py', *self.args(out), '--gate', self.gate, '--benchmark-steps', '3', '--benchmark-warmup', '1')
        report = json.loads((out/'benchmark.json').read_text())['benchmark']
        self.assertEqual(report['optimizer_steps'], 3)
        self.assertGreater(report['adapter_tensors_changed'], 0)
        self.assertFalse(report['test_evaluated'])
        self.assertEqual([x['epochs'] for x in report['eta']['scenarios']], [1, 2, 3])
        for name in ('test_scores.pt', 'gate.json', 'last_training.json'):
            self.assertFalse((out/name).exists())

    def test_cache_input_mutation_is_rejected(self):
        folder = self.root/'hash_fixture'
        folder.mkdir()
        (folder/'dataset.json').write_text('{}')
        (folder/'array.npy').write_bytes(b'original')
        (folder/'preparation.json').write_text(json.dumps(dict(array_sha256={'array.npy': sha256(folder/'array.npy')})))
        source_identity(folder)
        (folder/'array.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            source_identity(folder)

    def test_scaler_preserves_original_column_reduction_order(self):
        values = np.random.default_rng(7).normal(size=(10003, 4))
        original = np.asfortranarray(values)
        median, mean, sd = fit_risk_scaler(values)
        np.testing.assert_array_equal(median, np.nanmedian(original, axis=0))
        np.testing.assert_array_equal(mean, original.mean(axis=0))
        np.testing.assert_array_equal(sd, original.std(axis=0))
