"""Regression tests for review findings and the v1 contract."""
import copy
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import common
from gates import make_gate, validate_gate, official_targets, content_hash
from recipes import fit_risk_scaler, risk_features, record_epoch_order, record_improved, select_risk_candidate
from data import PrefixSplit, dataset_content_hash, multi_hot
from lora import LoRALinear


class GateRegressionTests(unittest.TestCase):
    def log(self):
        identity = dict(protocol_version=common.PROTOCOL_VERSION, task='record', disease=None,
                        smoke=False, model_sha256='model', dataset_metadata_sha256='metadata',
                        dataset_content_sha256='content', precision='float32',
                        validation_patient_ids_sha256='validation', test_patient_ids_sha256='test',
                        label_definition_sha256='labels', training_selection='all', n_train=2,
                        train_patient_ids_sha256='train')
        return dict(summary=dict(arm='head_only', optimizer_steps=1, selected=dict(epoch=1),
                                 test=dict(recall_at_10=55.68)),
                    fingerprint=dict(recipe='v1', gate_identity=identity))

    def test_alternate_targets_cannot_authorize_a_real_run(self):
        log = self.log()
        log['summary']['test']['recall_at_10'] = 0.
        targets = {'test.recall_at_10': dict(expected=0., tolerance=0.)}
        gate = make_gate(log, targets)
        self.assertTrue(gate['custom_targets'])
        self.assertFalse(gate['passed'])
        self.assertTrue(validate_gate(gate, log['fingerprint']['gate_identity']))
        # Exercise the original --targets CLI bypass, not only the helper.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'summary.json').write_text(json.dumps(log))
            (root / 'targets.json').write_text(json.dumps(dict(protocol_version=common.PROTOCOL_VERSION, record=targets)))
            result = subprocess.run([sys.executable, str(PACKAGE / 'check_gate.py'),
                                     '--summary', str(root / 'summary.json'), '--targets', str(root / 'targets.json'),
                                     '--out', str(root / 'gate.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(json.loads((root / 'gate.json').read_text())['custom_targets'])

    def test_validator_recomputes_checks_instead_of_trusting_flags(self):
        log = self.log()
        identity = log['fingerprint']['gate_identity']
        gate = make_gate(log, official_targets('record', None))
        self.assertEqual(validate_gate(gate, identity), [])
        for value in (0., float('nan'), float('inf')):
            bad = copy.deepcopy(gate)
            bad['checks'][0]['value'] = value
            self.assertTrue(validate_gate(bad, identity))
        bad = copy.deepcopy(gate)
        bad['checks'][0]['tolerance'] = 100.
        bad['targets_sha256'] = content_hash({'test.recall_at_10': dict(expected=55.68, tolerance=100.)})
        self.assertTrue(validate_gate(bad, identity))
        for field, value in (('optimizer_steps', 0), ('recipe', 'experimental')):
            bad = copy.deepcopy(gate)
            bad[field] = value
            self.assertTrue(validate_gate(bad, identity))

    def test_changed_pool_or_precision_invalidates_full_pool_gate(self):
        log = self.log()
        gate = make_gate(log, official_targets('record', None))
        for field in ('dataset_content_sha256', 'precision'):
            identity = dict(log['fingerprint']['gate_identity'], **{field: 'changed'})
            self.assertTrue(validate_gate(gate, identity))


class RecipeTests(unittest.TestCase):
    def test_published_c_is_fixed_even_when_smoke_testing(self):
        from recipes import risk_c_values, RISK_C_VALUES
        self.assertEqual(risk_c_values('v1'), (10.,))
        self.assertEqual(risk_c_values('experimental'), RISK_C_VALUES)
        for values, steps in (([1.], 200), ([.01, .1, 1., 10.], 200), ([10.], 199)):
            with self.assertRaisesRegex(ValueError, 'fixed C=10'):
                risk_c_values('v1', values, steps)

    def test_matching_metrics_do_not_hide_wrong_risk_selection(self):
        log = GateRegressionTests().log()
        identity = log['fingerprint']['gate_identity']
        identity.update(task='risk', disease='ckd')
        targets = official_targets('risk', 'ckd')
        log['summary']['test'] = {k.split('.')[1]: v['expected'] for k, v in targets.items()}
        log['summary']['selected']['C'] = 10.
        log['summary']['optimizer_steps'] = 200
        fit = dict(selection='fixed', candidates=[10.], steps_per_fit=200)
        log['fingerprint']['risk_regularization'] = fit
        self.assertTrue(make_gate(log, targets)['passed'])
        for change in (dict(selection='validation_brier'), dict(candidates=[1., 10.]), dict(steps_per_fit=199)):
            bad = copy.deepcopy(log)
            bad['fingerprint']['risk_regularization'].update(change)
            gate = make_gate(bad, targets)
            self.assertFalse(gate['passed'])
            gate['passed'] = True
            self.assertTrue(validate_gate(gate, identity))
        log['summary']['selected']['C'] = 1.
        self.assertFalse(make_gate(log, targets)['passed'])

    def test_lora_preserves_base_dtype_and_initial_gradient_direction(self):
        layer = LoRALinear(torch.nn.Linear(3, 2).double(), rank=2, alpha=2., dropout=0.)
        x = torch.ones(4, 3, dtype=torch.float64)
        torch.testing.assert_close(layer(x), layer.base(x), rtol=0, atol=0)
        layer(x).sum().backward()
        self.assertEqual(layer.lora_A.dtype, torch.float64)
        self.assertEqual(float(layer.lora_A.grad.abs().sum()), 0.)
        self.assertGreater(float(layer.lora_B.grad.abs().sum()), 0.)

    def test_v1_overrides_need_explicit_experimental_recipe(self):
        from recipes import configure_recipe
        parser = common.add_common_arguments(__import__('argparse').ArgumentParser())
        args = parser.parse_args(['--data', 'unused', '--out', 'unused', '--arm', 'head_only', '--lr-head', '.1'])
        cfg = common.resolve_defaults(args, dict(head_only=dict(epochs=60, patience=10, lr_head=.001, weight_decay=.01)))
        with self.assertRaisesRegex(ValueError, 'experimental'):
            configure_recipe(args, cfg, 'record')
        args.recipe = 'experimental'
        configure_recipe(args, cfg, 'record')

    def test_risk_preprocessing_reference_and_gradients(self):
        # Includes a censored eligible patient with a different distribution,
        # all-missing column, small-but-valid sd, zero sd and infinite input.
        x = np.array([[1., np.nan, 1e-5, 3., np.inf], [2., np.nan, 2e-5, 3., 2.],
                      [100., np.nan, 3e-5, 3., 4.]])
        median, mean, sd = fit_risk_scaler(x)
        np.testing.assert_allclose(median, [2., 0., 2e-5, 3., 4.])
        self.assertAlmostEqual(mean[0], 103. / 3)
        self.assertAlmostEqual(sd[2], np.std(x[:, 2]))
        np.testing.assert_array_equal(sd[[1, 3]], [1., 1.])
        z = np.array([[1., 3.], [2., 4.], [9., 8.]])
        pre = {}
        for name, values in (('clinical', x), ('embedding', z)):
            for key, value in zip(('median', 'mean', 'sd'), fit_risk_scaler(values)):
                pre[f'{name}_{key}'] = torch.from_numpy(value)
        embedding = torch.tensor(z, requires_grad=True)
        actual = risk_features(embedding, torch.tensor(x), pre)
        clean = np.where(np.isfinite(x), x, median)
        expected = np.concatenate([(clean - mean) / sd, (z - z.mean(0)) / z.std(0)], axis=1).astype(np.float32)
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=0, atol=1e-7)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(embedding.grad).all())
        self.assertGreater(float(embedding.grad.abs().sum()), 0.)

    def test_record_seed_order_and_tolerant_selection(self):
        for epoch in (0, 1, 4):
            np.testing.assert_array_equal(record_epoch_order(20, 20260908, epoch),
                                          np.random.default_rng(20260908 + epoch * 101).permutation(20))
        self.assertFalse(record_improved(50. + 1e-11, 3., (50., 2., 1)))
        self.assertTrue(record_improved(50. - 1e-11, 1., (50., 2., 1)))

    def test_risk_selection_uses_brier_then_smaller_c(self):
        candidates = [dict(C=10., validation_weighted_brier=.2), dict(C=1., validation_weighted_brier=.2),
                      dict(C=.01, validation_weighted_brier=.3)]
        self.assertEqual(select_risk_candidate(candidates), 1)

    def test_command_modules_are_safe_to_import(self):
        for name in ('train_record', 'train_risk', 'predict', 'check_gate', 'make_synthetic_dataset',
                     'prepare_record_v1', 'prepare_risk_baselines'):
            self.assertTrue(callable(importlib.import_module(name).main))

    def test_negative_label_is_not_silently_indexed_from_end(self):
        with self.assertRaises(ValueError):
            multi_hot([[-1]], np.arange(5), 5)


class DatasetTests(unittest.TestCase):
    def test_saved_prefix_mapping_and_cutoff(self):
        from prepare_record_v1 import pack_prefixes
        rows = np.array([[10, 100, 0, 1], [10, 105, 5, 2], [11, 80, 3, 1]], dtype=np.uint32)
        offsets = np.array([0, 2, 3])
        idx, age, typ = pack_prefixes(rows, offsets, [1, 0], [10, 11], [105, 80])
        np.testing.assert_array_equal(idx[:, :2], [[4, 0], [1, 6]])
        np.testing.assert_array_equal(age[:, :2], [[80, -10000], [100, 105]])
        np.testing.assert_array_equal(typ[:, :2], [[1, 0], [1, 2]])
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            pack_prefixes(rows, offsets, [0], [10, 11], [104, 80])
        with self.assertRaisesRegex(ValueError, 'identity'):
            pack_prefixes(rows, offsets, [0], [999, 11], [105, 80])

    def test_record_export_refuses_git_and_existing_output(self):
        from prepare_record_v1 import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.git').mkdir()
            with self.assertRaisesRegex(ValueError, 'Git checkout'):
                prepare(root, root, root / 'output')
            (root / '.git').rmdir()
            (root / 'output').mkdir()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(root, root, root / 'output')

    def test_eligibility_and_data_content_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run([sys.executable, str(PACKAGE / 'make_synthetic_dataset.py'), '--out', tmp,
                            '--patients', '8', '8', '8', '--max-length', '8'], check=True, capture_output=True)
            splits = {s: PrefixSplit(root, s, diseases=['ckd']) for s in ('train', 'validation', 'test')}
            before = dataset_content_hash(splits, 'risk', 'ckd')
            x = np.load(root / 'train/clinical.npy')
            x[0, 0] += 1
            np.save(root / 'train/clinical.npy', x)
            self.assertNotEqual(before, dataset_content_hash(splits, 'risk', 'ckd'))
            eligible = np.load(root / 'train/eligible_ckd.npy')
            observed = np.flatnonzero(splits['train'].outcomes['ckd'][1] > 0)[0]
            eligible[observed] = False
            np.save(root / 'train/eligible_ckd.npy', eligible)
            with self.assertRaisesRegex(ValueError, 'ineligible'):
                PrefixSplit(root, 'train', diseases=['ckd'])


if __name__ == '__main__':
    unittest.main()
