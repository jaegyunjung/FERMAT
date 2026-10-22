"""Unit and pipeline tests that need no release artifacts and no patient data."""
import argparse
import copy
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
from gates import content_hash, official_targets
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


class ObjectiveTests(unittest.TestCase):
    """Numeric checks requested in review: weighted objective and accumulation."""

    def accumulator(self, params, grad_accum, lr=1.):
        optimizer = torch.optim.SGD(params, lr=lr)
        args = argparse.Namespace(grad_accum=grad_accum, grad_clip=None, warmup_steps=0)
        return common.Accumulator(optimizer, lambda step: None, args)

    def test_weighted_objective_gradient_is_minus_0_4(self):
        y = torch.tensor([1., 0.]); w = torch.tensor([9., 1.])
        logit = torch.zeros(2, requires_grad=True)
        common.weighted_objective(logit, y, w).backward()
        self.assertAlmostEqual(float(logit.grad.sum()), -0.4, places=6)

    def test_one_patient_batches_reproduce_the_full_objective(self):
        y = torch.tensor([1., 0.]); w = torch.tensor([9., 1.])
        bias = torch.nn.Parameter(torch.zeros(1))
        acc = self.accumulator([bias], grad_accum=2)
        mean_weight = float(w.mean())
        for i in range(2):
            acc.backward(common.weighted_minibatch_loss(bias.expand(1), y[i:i + 1], w[i:i + 1], mean_weight), 1)
        acc.end_epoch()
        self.assertAlmostEqual(float(bias.detach()), 0.4, places=6)   # one SGD step of size -(-0.4)

    def test_partial_accumulation_uses_the_true_mean(self):
        p = torch.nn.Parameter(torch.zeros(1))
        acc = self.accumulator([p], grad_accum=4)
        for _ in range(2):
            acc.backward(2. * p, 1)
        acc.end_epoch()
        self.assertAlmostEqual(float(p.detach()), -2., places=6)
        self.assertEqual(acc.steps, 1)

    def test_unequal_batches_match_full_batch(self):
        torch.manual_seed(0)
        x = torch.randn(7, 3); y = (torch.rand(7) > 0.5).float(); w = torch.rand(7) + 0.5
        full = torch.nn.Linear(3, 1); split = copy.deepcopy(full)
        common.weighted_objective(full(x).squeeze(1), y, w).backward()
        acc = self.accumulator(list(split.parameters()), grad_accum=3)
        mean_weight = float(w.mean())
        captured = {}
        original_step = acc.optimizer.step
        acc.optimizer.step = lambda: captured.update(grads=[p.grad.clone() for p in split.parameters()]) or original_step()
        for rows in (slice(0, 3), slice(3, 5), slice(5, 7)):
            acc.backward(common.weighted_minibatch_loss(split(x[rows]).squeeze(1), y[rows], w[rows], mean_weight), rows.stop - rows.start)
        acc.end_epoch()
        for g_full, g_split in zip([p.grad for p in full.parameters()], captured['grads']):
            torch.testing.assert_close(g_full, g_split, rtol=1e-5, atol=1e-6)
        # Same check for the record loss (mean over patients of a per-patient sum).
        full = torch.nn.Linear(3, 4); split = copy.deepcopy(full); t = (torch.rand(7, 4) > 0.5).float()
        common.record_loss(full(x), t).backward()
        acc = self.accumulator(list(split.parameters()), grad_accum=2)
        captured = {}
        original_step = acc.optimizer.step
        acc.optimizer.step = lambda: captured.update(grads=[p.grad.clone() for p in split.parameters()]) or original_step()
        for rows in (slice(0, 4), slice(4, 7)):
            acc.backward(common.record_loss(split(x[rows]), t[rows]), rows.stop - rows.start)
        acc.end_epoch()
        for g_full, g_split in zip([p.grad for p in full.parameters()], captured['grads']):
            torch.testing.assert_close(g_full, g_split, rtol=1e-5, atol=1e-6)

    def test_zero_update_epoch_fails(self):
        acc = self.accumulator([torch.nn.Parameter(torch.zeros(1))], grad_accum=2)
        with self.assertRaises(RuntimeError):
            acc.end_epoch()


class GateValidationTests(unittest.TestCase):
    def identity(self, **overrides):
        base = dict(protocol_version=common.PROTOCOL_VERSION, task='risk', disease='ckd', smoke=False,
                    model_sha256='m', dataset_metadata_sha256='d', validation_patient_ids_sha256='v',
                    test_patient_ids_sha256='t', label_definition_sha256='l', training_selection='all',
                    n_train=10, train_patient_ids_sha256='tr', dataset_content_sha256='content', precision='float32')
        base.update(overrides)
        return base

    def gate(self, identity, metrics=('test.weighted_auroc', 'test.weighted_brier'), **overrides):
        targets = official_targets('risk', 'ckd')
        checked_targets = {m: targets[m] for m in metrics}
        g = dict(passed=True, custom_targets=False, identity=identity, recipe='v1',
                 optimizer_steps=200, selected_epoch=1, targets_sha256=content_hash(checked_targets),
                 risk_regularization=dict(selection='fixed', candidates=[10.0], steps_per_fit=200, selected_C=10.0),
                 checks=[dict(metric=m, value=targets[m]['expected'], **targets[m], passed=True) for m in metrics])
        g.update(overrides)
        return g

    def test_matching_gate_is_accepted(self):
        self.assertEqual(common.validate_gate(self.gate(self.identity()), self.identity()), [])

    def test_full_pool_gate_covers_a_subset_but_not_the_reverse(self):
        run = self.identity(training_selection='patient_list', n_train=5, train_patient_ids_sha256='sub')
        self.assertEqual(common.validate_gate(self.gate(self.identity()), run), [])
        gate_on_subset = self.gate(self.identity(training_selection='subset', train_patient_ids_sha256='other'))
        self.assertTrue(any('training patients' in r for r in common.validate_gate(gate_on_subset, run)))

    def test_mismatches_are_rejected(self):
        run = self.identity()
        for field in ('task', 'disease', 'model_sha256', 'test_patient_ids_sha256', 'label_definition_sha256',
                      'protocol_version', 'smoke'):
            gate = self.gate(self.identity(**{field: 'x' if field != 'smoke' else True}))
            self.assertTrue(any(r.startswith(field) for r in common.validate_gate(gate, run)), field)
        self.assertTrue(common.validate_gate(self.gate(self.identity(), metrics=('test.weighted_brier',)), run))
        self.assertTrue(common.validate_gate(self.gate(self.identity(), custom_targets=True), run))
        self.assertEqual(common.validate_gate(self.gate(self.identity(smoke=True), custom_targets=True), self.identity(smoke=True)), [])
        self.assertTrue(common.validate_gate(self.gate(self.identity(), passed=False), run))


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

    def gate_for(self, summary_path, metrics, out, offset=0., complete=True):
        summary = json.loads(Path(summary_path).read_text())['summary']
        command = ['check_gate.py', '--summary', str(summary_path), '--out', str(out)]
        for metric in metrics:
            value = summary
            for key in metric.split('.'):
                value = value[key]
            command += ['--metric', metric, '--expected', str(value + offset), '--tolerance', '1e-9']
        should_pass = offset == 0. and complete
        self.invoke(command, expect_failure=not should_pass)
        gate = json.loads(Path(out).read_text())
        self.assertEqual(gate['passed'], should_pass)
        self.assertTrue(gate['custom_targets'])
        return out

    def test_a_record_task_both_arms_with_gate_and_reload(self):
        head = self.root / 'record_head'
        self.invoke(['train_record.py'] + self.common_args(head, 'head_only'))
        head_summary = json.loads((head / 'summary.json').read_text())
        self.assertEqual(head_summary['fingerprint']['warmup_steps'], 0)
        self.assertGreater(head_summary['summary']['optimizer_steps'], 0)
        gate = self.gate_for(head / 'summary.json', ['test.recall_at_10'], self.root / 'record_gate.json')
        self.gate_for(head / 'summary.json', ['test.recall_at_10'], self.root / 'record_gate_fail.json', offset=5.)
        protocol_gate = self.invoke(['check_gate.py', '--summary', str(head / 'summary.json'),
                                     '--out', str(self.root / 'record_gate_protocol.json')], expect_failure=True)
        self.assertIn('"custom_targets": false', protocol_gate)   # real targets, synthetic data: fails honestly
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

    def test_benchmark_trains_adapters_without_test_scoring_or_gate(self):
        head = self.root / 'benchmark_head'
        self.invoke(['train_record.py'] + self.common_args(head, 'head_only'))
        gate = self.gate_for(head / 'summary.json', ['test.recall_at_10'], self.root / 'benchmark_gate.json')
        out = self.root / 'benchmark'
        self.invoke(['train_record.py'] + self.common_args(out, 'lora') +
                    ['--gate', str(gate), '--benchmark-steps', '4', '--benchmark-warmup', '1'])
        result = json.loads((out / 'benchmark.json').read_text())['benchmark']
        self.assertEqual(result['optimizer_steps'], 4)
        self.assertEqual(result['patients_processed'], 32)
        self.assertTrue(result['audit']['frozen_base_unchanged'])
        self.assertGreater(result['adapter_tensors_changed'], 0)
        self.assertFalse(result['test_evaluated'])
        for name in ('summary.json', 'gate.json', 'test_scores.pt', 'test_patient_metrics.csv'):
            self.assertFalse((out / name).exists())

    def test_experimental_control_uses_explicit_optimizer_batch(self):
        from types import SimpleNamespace
        from run_record_suite import verify_run
        out = self.root / 'matched_control'
        baseline = self.root / 'suite_baseline.csv'
        ids = np.load(self.data / 'test/patient_id.npy')
        baseline.write_text('patient_id,recall_at_10\n' + ''.join(f'{int(pid)},0\n' for pid in ids))
        self.invoke(['train_record.py'] + self.common_args(out, 'head_only') +
                    ['--recipe', 'experimental', '--optimizer-batch-size', '8',
                     '--warmup-steps', '100', '--grad-clip', '1', '--baseline-recall', str(baseline),
                     '--baseline-label', 'generation50', '--bootstrap-resamples', '5000'])
        log = json.loads((out / 'summary.json').read_text())
        self.assertEqual(log['fingerprint']['resolved']['batch'], 8)
        self.assertEqual(log['summary']['optimizer_steps'], 10)  # 40 / 8 * 2 epochs
        self.assertEqual(log['fingerprint']['warmup_steps'], 100)
        self.assertEqual(log['fingerprint']['grad_clip'], 1.)
        self.assertFalse((out / 'gate.json').exists())
        gate = self.gate_for(out / 'summary.json', ['test.recall_at_10'], self.root / 'suite_gate.json')
        # Only this synthetic fixture replaces disabled tracking with a successful fake delivery.
        (out / 'tracking/tracking.json').write_text(json.dumps(dict(connected=True, success=True, run_id='synthetic')))
        job = dict(batch=8, epochs=2, patience=2, warmup=100, clip=1., seed=log['fingerprint']['arguments']['seed'])
        args = SimpleNamespace(gate=gate, data=self.data, expected_commit=log['fingerprint']['git_commit'], baseline_recall=baseline)
        self.assertTrue(verify_run(out, job, args)['passed'])
        log['summary']['test']['recall_at_10'] += .1
        (out / 'summary.json').write_text(json.dumps(log))
        with self.assertRaisesRegex(ValueError, 'Summary metric differs'):
            verify_run(out, job, args)
        for recipe, arm, size in [('v1', 'head_only', '8'), ('lora', 'lora', '8'),
                                  ('experimental', 'head_only', '0')]:
            error = self.invoke(['train_record.py'] + self.common_args(self.root / 'invalid_control', arm) +
                                ['--recipe', recipe, '--optimizer-batch-size', size], expect_failure=True)
            self.assertIn('--optimizer-batch-size requires', error)

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
        out = self.root / 'record_missing_ids'
        message = self.invoke(['train_record.py'] + self.common_args(out, 'head_only') + ['--train-patient-ids', str(listed)],
                              expect_failure=True)
        self.assertIn('not in the training split', message)

    def test_epoch_resume_matches_uninterrupted_and_reloads_validation(self):
        import checkpointing
        def same(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    same(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    same(a, b)
            else:
                self.assertEqual(left, right)
        gate_head = self.root / 'resume_gate_head'
        self.invoke(['train_record.py'] + self.common_args(gate_head, 'head_only'))
        gate = self.gate_for(gate_head / 'summary.json', ['test.recall_at_10'], self.root / 'resume_gate.json')
        for arm in ('head_only', 'lora'):
            with self.subTest(arm=arm):
                full, pause, resumed = [self.root / (arm + '_resume_' + suffix) for suffix in ('full', 'pause', 'continued')]
                def args(out):
                    return self.common_args(out, arm) + ['--recipe', 'lora' if arm == 'lora' else 'experimental',
                        '--validation-only', '--seed', '3', '--grad-accum', '2', '--gate', str(gate)]
                self.invoke(['train_record.py'] + args(full))
                self.invoke(['train_record.py'] + args(pause) + ['--stop-after-epoch', '1'])
                self.assertTrue((pause / 'paused.json').exists())
                self.assertFalse((pause / 'test_scores.pt').exists())
                self.invoke(['train_record.py'] + args(resumed) + ['--resume', str(pause / 'last_training.json')])
                def latest(folder):
                    manifest = json.loads((folder / 'last_training.json').read_text())
                    return torch.load(folder / manifest['checkpoint'], weights_only=True)
                a, b = latest(full), latest(resumed)
                for folder, state in ((full, a), (resumed, b)):
                    history = [json.loads(line) for line in
                               (folder / 'tracking/aggregate_history.jsonl').read_text().splitlines()]
                    updates = [row['progress/optimizer_steps'] for row in history
                               if 'progress/optimizer_steps' in row]
                    self.assertEqual(updates[-1], state['optimizer_steps'])
                for field in ('current_state', 'optimizer', 'best_state', 'best', 'since_best', 'optimizer_steps', 'rng'):
                    same(a[field], b[field])
                for epoch_a, epoch_b in zip(a['epochs'], b['epochs']):
                    same({k:v for k,v in epoch_a.items() if k != 'epoch_seconds'},
                         {k:v for k,v in epoch_b.items() if k != 'epoch_seconds'})
                same(torch.load(full / 'validation_scores.pt', weights_only=True),
                     torch.load(resumed / 'validation_scores.pt', weights_only=True))
                reload_out = self.root / (arm + '_resume_reload')
                self.invoke(['predict.py', '--state', str(resumed / 'trainable_state.pt'), '--data', str(self.data),
                             '--out', str(reload_out), '--device', 'cpu', '--split', 'validation', '--batch-size', '8',
                             '--reference-scores', str(resumed / 'validation_scores.pt'),
                             '--smoke-random-init', str(HERE / 'smoke_model_config.json')])
                self.assertTrue(json.loads((reload_out / 'reload_audit.json').read_text())['passed'])
                self.assertFalse((resumed / 'gate.json').exists())
                error = self.invoke(['train_record.py'] + args(self.root / (arm + '_wrong_seed')) +
                                    ['--resume', str(pause / 'last_training.json'), '--seed', '4'], expect_failure=True)
                self.assertIn('Resume contract mismatch', error)
                # Resuming an epoch-cap checkpoint finalizes; it must not execute another epoch.
                terminal_out = self.root / (arm + '_terminal_resume')
                self.invoke(['train_record.py'] + args(terminal_out) + ['--resume', str(resumed / 'last_training.json')])
                summary = json.loads((terminal_out / 'validation_summary.json').read_text())
                self.assertEqual(len(summary['epochs']), 2)
                self.assertEqual(summary['summary']['optimizer_steps'], a['optimizer_steps'])
    def test_split_overlap_is_rejected(self):
        from data import PrefixSplit
        import shutil
        copy = self.root / 'overlap'
        shutil.copytree(self.data, copy)
        np.save(copy / 'validation/patient_id.npy', np.load(copy / 'train/patient_id.npy')[:16])
        splits = {name: PrefixSplit(copy, name) for name in ('train', 'validation', 'test')}
        with self.assertRaises(ValueError):
            common.check_split_overlap(splits)

    def test_b_risk_task_both_arms_with_gate_and_reload(self):
        head = self.root / 'risk_head'
        self.invoke(['train_risk.py', '--disease', 'ckd'] + self.common_args(head, 'head_only'))
        head_log = json.loads((head / 'summary.json').read_text())
        self.assertEqual([c['C'] for c in head_log['candidates']], [10.])
        self.assertEqual(head_log['summary']['optimizer_steps'], 200)
        eligible = np.load(self.data / 'train/eligible_ckd.npy')
        self.assertEqual(head_log['fingerprint']['n_preprocessing_train_eligible'], int(eligible.sum()))
        clinical = np.load(self.data / 'train/clinical.npy').astype(np.float64)
        state = torch.load(head / 'trainable_state.pt', weights_only=True)
        np.testing.assert_allclose(state['preprocessing']['clinical_mean'], clinical[eligible].mean(0), rtol=0, atol=1e-12)
        incomplete = self.gate_for(head / 'summary.json', ['test.weighted_brier'], self.root / 'risk_gate_half.json', complete=False)
        self.assertFalse(json.loads(Path(incomplete).read_text())['passed'])   # AUROC check missing
        gate = self.gate_for(head / 'summary.json', ['test.weighted_auroc', 'test.weighted_brier'], self.root / 'risk_gate.json')
        lora = self.root / 'risk_lora'
        wrong_task = self.invoke(['train_risk.py', '--disease', 'ckd'] + self.common_args(lora, 'lora') +
                                 ['--gate', str(self.root / 'record_gate.json')], expect_failure=True)
        self.assertIn('task:', wrong_task)
        self.invoke(['train_risk.py', '--disease', 'ckd'] + self.common_args(lora, 'head_only') + ['--grad-accum', '2'],
                    expect_failure=True)
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
        # Different inference batch sizes can change float32 rounding. Compare
        # identifiers/labels/weights exactly, probabilities within one micro-unit.
        a = np.genfromtxt(lora / 'test_patient_probabilities.csv', delimiter=',', names=True)
        b = np.genfromtxt(reload / 'test_patient_probabilities.csv', delimiter=',', names=True)
        for column in ('patient_id', 'outcome', 'weight'):
            np.testing.assert_array_equal(a[column], b[column])
        np.testing.assert_allclose(a['probability'], b['probability'], rtol=0, atol=1e-6)

    def test_existing_run_cannot_be_overwritten(self):
        out = self.root / 'protected_run'
        out.mkdir()
        sentinel = out / 'summary.json'
        sentinel.write_text('existing result')
        message = self.invoke(['train_record.py'] + self.common_args(out, 'head_only'), expect_failure=True)
        self.assertIn('not empty', message)
        self.assertEqual(sentinel.read_text(), 'existing result')

    def test_checkpoint_rejects_another_base_model(self):
        head = self.root / 'mismatch_head'
        self.invoke(['train_record.py'] + self.common_args(head, 'head_only'))
        state = torch.load(head / 'trainable_state.pt', weights_only=True)
        state['config']['model_sha256'] = 'different-model'
        wrong = self.root / 'wrong.pt'
        torch.save(state, wrong)
        message = self.invoke(['predict.py', '--state', str(wrong), '--data', str(self.data),
                               '--out', str(self.root / 'wrong_prediction'), '--device', 'cpu',
                               '--smoke-random-init', str(HERE / 'smoke_model_config.json')], expect_failure=True)
        self.assertIn('Base model does not match', message)


if __name__ == '__main__':
    unittest.main()
