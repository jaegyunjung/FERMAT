"""Unit and pipeline tests that need no release artifacts and no patient data."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(PACKAGE))
import common  # noqa: E402  (also puts the release package on sys.path)
from fermat_release.model import Fermat, FermatConfig  # noqa: E402
from lora import (LoRALinear, inject_lora, lora_parameters, lora_state_dict, load_lora_state_dict,  # noqa: E402
                  count_parameters, snapshot_parameters, assert_base_unchanged)
from metrics import record_metrics, weighted_auroc, weighted_brier, paired_bootstrap, mean_difference  # noqa: E402
from data import class_lookup, multi_hot, fixed_subset  # noqa: E402

CONFIG = json.loads((HERE / 'smoke_model_config.json').read_text())


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return Fermat(FermatConfig(**CONFIG)).eval()


def tiny_batch():
    idx = torch.tensor([[3, 20, 21, 22, 23, 0], [4, 30, 31, 0, 0, 0]])
    age = torch.tensor([[0., 100., 100., 130., 500., -10000.], [0., 50., 80., -10000., -10000., -10000.]])
    typ = torch.tensor([[7, 1, 2, 3, 1, 0], [7, 1, 1, 0, 0, 0]])
    return idx, age, typ


class LoraTests(unittest.TestCase):
    def test_injection_is_identity_until_trained(self):
        model = tiny_model()
        idx, age, typ = tiny_batch()
        with torch.no_grad():
            before = common.last_hidden(model, idx, age, typ).clone()
            logits_before, _, _ = model(idx, age, typ, return_attention=False)
        wrapped = inject_lora(model, rank=4, alpha=8., dropout=0.0, targets='attention')
        self.assertEqual(len(wrapped), 2 * CONFIG['n_layer'])
        with torch.no_grad():
            after = common.last_hidden(model, idx, age, typ)
            logits_after, _, _ = model(idx, age, typ, return_attention=False)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)

    def test_trainable_parameters_and_frozen_base(self):
        model = tiny_model()
        snapshot = snapshot_parameters(model)
        total_before = count_parameters(model)['total']
        inject_lora(model, rank=4, alpha=8., dropout=0.0, targets='attention')
        counts = count_parameters(model)
        d = CONFIG['n_embd']
        expected = CONFIG['n_layer'] * (4 * (d + 3 * d) + 4 * (d + d))
        self.assertEqual(counts['trainable'], expected)
        self.assertEqual(counts['total'], total_before + expected)
        self.assertTrue(all(p.requires_grad for p in lora_parameters(model)))
        self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n))
        idx, age, typ = tiny_batch()
        model.train()
        optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
        loss = common.last_hidden(model, idx, age, typ).pow(2).mean()
        loss.backward()
        optimizer.step()
        assert_base_unchanged(model, snapshot)
        self.assertTrue(any(float(p.detach().abs().sum()) > 0 for n, p in model.named_parameters() if 'lora_B' in n))
        with self.assertRaises(ValueError):
            inject_lora(model, rank=4)

    def test_attention_mlp_targets_and_merge(self):
        model = tiny_model()
        wrapped = inject_lora(model, rank=2, alpha=2., dropout=0.0, targets='attention_mlp')
        self.assertEqual(len(wrapped), 4 * CONFIG['n_layer'])
        layer = model.transformer.h[0].attn.c_attn
        self.assertIsInstance(layer, LoRALinear)
        with torch.no_grad():
            layer.lora_B.normal_()
            x = torch.randn(3, CONFIG['n_embd'])
            merged = torch.nn.functional.linear(x, layer.merged_weight())
            torch.testing.assert_close(layer(x), merged, rtol=1e-5, atol=1e-5)

    def test_state_roundtrip(self):
        model = tiny_model()
        inject_lora(model, rank=2, alpha=2., dropout=0.0)
        with torch.no_grad():
            for p in lora_parameters(model):
                p.normal_()
        state = lora_state_dict(model)
        other = tiny_model(seed=1)
        inject_lora(other, rank=2, alpha=2., dropout=0.0)
        load_lora_state_dict(other, state)
        for a, b in zip(lora_parameters(model), lora_parameters(other)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        third = tiny_model(seed=2)
        inject_lora(third, rank=3, alpha=2., dropout=0.0)
        with self.assertRaises(ValueError):
            load_lora_state_dict(third, state)


class MetricTests(unittest.TestCase):
    def test_record_metrics(self):
        classes = np.array([10, 11, 12, 13, 14])
        index = class_lookup(classes, 20)
        scores = torch.tensor([[0.9, 0.8, 0.1, 0.2, 0.3], [0.1, 0.1, 0.1, 0.1, 0.1]])
        labels = [np.array([10, 13]), np.array([14])]
        out = record_metrics(scores, labels, index, ks=(1, 2))
        np.testing.assert_allclose(out['recall_at_1'], [50., 0.])
        np.testing.assert_allclose(out['recall_at_2'], [50., 0.])   # stable tie order: classes 10, 11 first
        np.testing.assert_allclose(out['precision_at_2'], [50., 0.])
        np.testing.assert_allclose(out['f1_at_2'], [50., 0.])
        np.testing.assert_allclose(out['hit_at_2'], [100., 0.])
        target = multi_hot(labels, index, 5)
        self.assertEqual(target.sum().item(), 3)
        with self.assertRaises(ValueError):
            multi_hot([np.array([19])], index, 5)

    def test_weighted_auroc_and_brier(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.2, 0.4, 0.6])
        w = np.ones(4)
        self.assertAlmostEqual(weighted_auroc(y, p, w), 0.75)
        self.assertAlmostEqual(weighted_auroc(y, np.array([0.5] * 4), w), 0.5)
        w2 = np.array([1., 1., 3., 1.])
        # pairs: (0.9>0.2),(0.9>0.6) weight 1 each; (0.4>0.2) weight 3; (0.4<0.6) 0
        self.assertAlmostEqual(weighted_auroc(y, p, w2), (1 + 1 + 3) / (4 * 2))
        self.assertAlmostEqual(weighted_brier(y, p, w), np.mean((p - y) ** 2))
        self.assertTrue(np.isnan(weighted_auroc(np.array([1, 1]), np.array([.2, .3]), np.ones(2))))

    def test_paired_bootstrap_is_paired(self):
        a = np.arange(100, dtype=float)
        result = paired_bootstrap(mean_difference, [a + 2., a], 200, seed=0)
        self.assertAlmostEqual(result['point'], 2.)
        self.assertAlmostEqual(result['ci_low'], 2.)
        self.assertAlmostEqual(result['ci_high'], 2.)

    def test_fixed_subset_is_deterministic(self):
        np.testing.assert_array_equal(fixed_subset(50, 10, 3), fixed_subset(50, 10, 3))
        self.assertEqual(len(fixed_subset(50, None, 3)), 50)


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = cls.root / 'data'
        cls.invoke(['make_synthetic_dataset.py', '--out', str(cls.data), '--patients', '40', '16', '16',
                    '--max-length', '24', '--n-classes', '30'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def invoke(cls, arguments, expect_failure=False):
        completed = subprocess.run([sys.executable, str(PACKAGE / arguments[0])] + arguments[1:],
                                   capture_output=True, text=True, cwd=PACKAGE)
        if expect_failure:
            if completed.returncode == 0:
                raise AssertionError('expected failure but the command succeeded:\n' + completed.stdout)
            return completed.stderr + completed.stdout
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        return completed.stdout

    def common_args(self, out, arm):
        return ['--data', str(self.data), '--out', str(out), '--arm', arm, '--device', 'cpu',
                '--smoke-random-init', str(HERE / 'smoke_model_config.json'), '--batch-size', '8',
                '--epochs', '2', '--patience', '2', '--bootstrap-resamples', '20']

    def gate_for(self, summary_path, metric, out):
        value = json.loads(Path(summary_path).read_text())['summary']
        for key in metric.split('.'):
            value = value[key]
        self.invoke(['check_gate.py', '--summary', str(summary_path), '--metric', metric,
                     '--expected', str(value), '--tolerance', '1e-9', '--out', str(out)])
        self.assertTrue(json.loads(Path(out).read_text())['passed'])
        return out

    def test_record_task_both_arms_with_gate_and_reload(self):
        head = self.root / 'record_head'
        self.invoke(['train_record.py'] + self.common_args(head, 'head_only'))
        head_summary = json.loads((head / 'summary.json').read_text())
        self.assertEqual(head_summary['fingerprint']['warmup_steps'], 0)
        self.assertGreater(head_summary['summary']['optimizer_steps'], 0)
        gate = self.gate_for(head / 'summary.json', 'test.recall_at_10', self.root / 'record_gate.json')
        failing = self.invoke(['check_gate.py', '--summary', str(head / 'summary.json'), '--metric',
                               'test.recall_at_10', '--expected', '999', '--tolerance', '0.1',
                               '--out', str(self.root / 'record_gate_fail.json')], expect_failure=True)
        self.assertIn('"passed": false', failing)
        lora = self.root / 'record_lora'
        ungated = self.invoke(['train_record.py'] + self.common_args(lora, 'lora'), expect_failure=True)
        self.assertIn('requires --gate', ungated)
        # Five training batches of eight patients with accumulation six: the only
        # update per epoch comes from the trailing partial accumulation.
        self.invoke(['train_record.py'] + self.common_args(lora, 'lora') +
                    ['--gate', str(gate), '--grad-accum', '6', '--train-subset', '40',
                     '--baseline-recall', str(head / 'test_patient_metrics.csv'), '--baseline-label', 'head_only'])
        log = json.loads((lora / 'summary.json').read_text())
        summary = log['summary']
        self.assertEqual(summary['optimizer_steps'], len(log['epochs']))
        self.assertTrue(summary['audit']['frozen_base_unchanged'])
        self.assertEqual(summary['audit']['adapter_parameters'], 2 * CONFIG['n_layer'] * (8 * (4 * CONFIG['n_embd']) + 8 * (2 * CONFIG['n_embd'])) // 2)
        self.assertEqual(summary['audit']['trained_parameters_total'],
                         summary['audit']['adapter_parameters'] + summary['audit']['head_parameters'])
        self.assertIn('paired_recall_at_10_minus_head_only', summary)
        # Reload in a separate process and reproduce the test predictions exactly.
        reload = self.root / 'record_reload'
        self.invoke(['predict.py', '--state', str(lora / 'trainable_state.pt'), '--data', str(self.data),
                     '--out', str(reload), '--device', 'cpu', '--smoke-random-init', str(HERE / 'smoke_model_config.json')])
        a = torch.load(lora / 'test_scores.pt', weights_only=True)
        b = torch.load(reload / 'test_scores.pt', weights_only=True)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=1e-5, atol=1e-5)
        self.assertEqual((lora / 'test_patient_metrics.csv').read_text(), (reload / 'test_patient_metrics.csv').read_text())

    def test_training_selection_is_exact(self):
        out = self.root / 'record_toolarge'
        message = self.invoke(['train_record.py'] + self.common_args(out, 'head_only') + ['--train-subset', '41'],
                              expect_failure=True)
        self.assertIn('exceeds', message)
        ids = np.load(self.data / 'train/patient_id.npy')[[3, 1, 7, 20]]
        listed = self.root / 'listed_ids.npy'
        np.save(listed, ids)
        out = self.root / 'record_listed'
        self.invoke(['train_record.py'] + self.common_args(out, 'head_only') + ['--train-patient-ids', str(listed)])
        fingerprint = json.loads((out / 'summary.json').read_text())['fingerprint']
        self.assertEqual(fingerprint['n_train'], 4)
        self.assertEqual(fingerprint['training_selection'], 'patient_list')
        np.save(listed, np.append(ids, 99999999))
        message = self.invoke(['train_record.py'] + self.common_args(out, 'head_only') + ['--train-patient-ids', str(listed)],
                              expect_failure=True)
        self.assertIn('not in the training split', message)

    def test_split_overlap_is_rejected(self):
        from data import PrefixSplit
        import shutil
        copy = self.root / 'overlap'
        shutil.copytree(self.data, copy)
        np.save(copy / 'validation/patient_id.npy', np.load(copy / 'train/patient_id.npy')[:16])
        splits = {name: PrefixSplit(copy, name) for name in ('train', 'validation', 'test')}
        with self.assertRaises(ValueError):
            common.check_split_overlap(splits)

    def test_risk_task_both_arms_with_gate_and_reload(self):
        head = self.root / 'risk_head'
        self.invoke(['train_risk.py', '--disease', 'ckd'] + self.common_args(head, 'head_only'))
        gate = self.gate_for(head / 'summary.json', 'test.weighted_brier', self.root / 'risk_gate.json')
        lora = self.root / 'risk_lora'
        self.invoke(['train_risk.py', '--disease', 'ckd'] + self.common_args(lora, 'lora') +
                    ['--gate', str(gate), '--grad-accum', '2',
                     '--baseline-probabilities', str(head / 'test_patient_probabilities.csv')])
        summary = json.loads((lora / 'summary.json').read_text())['summary']
        self.assertTrue(summary['audit']['frozen_base_unchanged'])
        self.assertGreater(summary['optimizer_steps'], 0)
        self.assertIn('auroc', summary['paired_minus_baseline'])
        reload = self.root / 'risk_reload'
        self.invoke(['predict.py', '--state', str(lora / 'trainable_state.pt'), '--data', str(self.data),
                     '--out', str(reload), '--device', 'cpu', '--smoke-random-init', str(HERE / 'smoke_model_config.json')])
        self.assertEqual((lora / 'test_patient_probabilities.csv').read_text(),
                         (reload / 'test_patient_probabilities.csv').read_text())


if __name__ == '__main__':
    unittest.main()
