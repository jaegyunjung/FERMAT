import unittest

import torch

from model import (
    Fermat,
    FermatConfig,
    TokenType,
    align_time_deltas,
    build_attention_mask,
    build_target_mask,
)


class TieMaskTest(unittest.TestCase):
    def test_same_date_events_are_hidden_from_attention(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        age = torch.tensor([[1.0, 2.0, 2.0, 4.0]])
        target_age = torch.tensor([[2.0, 2.0, 4.0, 7.0]])

        mask = build_attention_mask(
            idx,
            age,
            targets_age=target_age,
            mask_ties=True,
        )

        self.assertEqual(mask.shape, (1, 1, 4, 4))
        self.assertTrue(mask[0, 0, 1, 0])
        self.assertFalse(mask[0, 0, 1, 1])
        self.assertFalse(mask[0, 0, 1, 2])

    def test_time_delta_uses_latest_visible_non_tied_event(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        age = torch.tensor([[1.0, 2.0, 2.0, 4.0]])
        target_age = torch.tensor([[2.0, 2.0, 4.0, 7.0]])
        mask = build_attention_mask(
            idx,
            age,
            targets_age=target_age,
            mask_ties=True,
        )

        dt = align_time_deltas(age, target_age, mask, mask_ties=True)

        torch.testing.assert_close(dt, torch.tensor([[1.0, 1.0, 2.0, 3.0]]))

    def test_time_delta_preserves_single_token_sequence_dimension(self):
        idx = torch.tensor([[1], [2]])
        age = torch.tensor([[1.0], [5.0]])
        target_age = torch.tensor([[2.0], [8.0]])
        mask = build_attention_mask(
            idx,
            age,
            targets_age=target_age,
            mask_ties=True,
        )

        dt = align_time_deltas(age, target_age, mask, mask_ties=True)

        self.assertEqual(dt.shape, (2, 1))
        torch.testing.assert_close(dt, torch.tensor([[1.0], [3.0]]))

    def test_time_delta_handles_empty_sequence(self):
        age = torch.empty((2, 0))
        target_age = torch.empty((2, 0))
        mask = torch.empty((2, 1, 0, 0), dtype=torch.bool)

        dt = align_time_deltas(age, target_age, mask, mask_ties=True)

        self.assertEqual(dt.shape, (2, 0))

    def test_padding_positions_keep_a_safe_diagonal(self):
        idx = torch.tensor([[0, 1]])
        age = torch.tensor([[-10000.0, 10.0]])
        target_age = torch.tensor([[-10000.0, 20.0]])

        mask = build_attention_mask(
            idx,
            age,
            targets_age=target_age,
            mask_ties=True,
        )

        self.assertTrue(mask[0, 0, 0, 0])


class TargetMaskTest(unittest.TestCase):
    def test_lab_context_mask_excludes_lab_targets(self):
        targets = torch.tensor([2, 3, 4, 5])
        target_types = torch.tensor([
            TokenType.DX,
            TokenType.LAB,
            TokenType.RX,
            TokenType.PAD,
        ])

        mask = build_target_mask(
            targets,
            target_types,
            ignore_tokens=[0],
            ignore_types=[TokenType.PAD, TokenType.LAB],
        )

        torch.testing.assert_close(
            mask,
            torch.tensor([True, False, True, False]),
        )

    def test_all_ignored_targets_produce_finite_zero_losses(self):
        config = FermatConfig(
            block_size=4,
            vocab_size=16,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            mask_ties=True,
            output_ignore_tokens=[0, 1],
            ignore_types=[
                TokenType.PAD,
                TokenType.SEX,
                TokenType.NO_EVENT,
                TokenType.LAB,
            ],
        )
        model = Fermat(config)
        idx = torch.tensor([[2, 3, 4, 5]])
        age = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
        token_type = torch.full_like(idx, TokenType.LAB)
        targets = torch.tensor([[3, 4, 5, 6]])
        targets_age = torch.tensor([[11.0, 12.0, 13.0, 14.0]])
        target_type = torch.full_like(targets, TokenType.LAB)

        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            targets_age,
            target_token_type=target_type,
        )
        combined = loss["loss_ce"] + loss["loss_dt"]
        combined.backward()

        self.assertEqual(int(loss["n_targets"]), 0)
        self.assertTrue(torch.isfinite(loss["loss_ce"]))
        self.assertTrue(torch.isfinite(loss["loss_dt"]))
        self.assertEqual(float(loss["loss_ce"].detach()), 0.0)
        self.assertEqual(float(loss["loss_dt"].detach()), 0.0)

    def test_output_ignore_tokens_are_masked_during_training(self):
        config = FermatConfig(
            block_size=2,
            vocab_size=8,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            output_ignore_tokens=[0, 1],
        )
        model = Fermat(config)
        idx = torch.tensor([[2, 3]])
        age = torch.tensor([[10.0, 20.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4]])
        target_age = torch.tensor([[20.0, 30.0]])

        logits, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            target_age,
            target_token_type=token_type,
        )

        self.assertTrue(torch.isneginf(logits[..., 0]).all())
        self.assertTrue(torch.isneginf(logits[..., 1]).all())
        self.assertTrue(torch.isfinite(loss["loss_ce"]))
        self.assertTrue(torch.isfinite(loss["loss_dt"]))

    def test_ce_only_forward_skips_unstable_time_loss(self):
        config = FermatConfig(
            block_size=2,
            vocab_size=8,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            t_min=0.0,
        )
        model = Fermat(config)
        idx = torch.tensor([[2, 3]])
        age = torch.tensor([[0.0, 1.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4]])
        target_age = torch.tensor([[1e30, 1e30]])

        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            target_age,
            target_token_type=token_type,
            compute_time_loss=False,
        )

        self.assertTrue(torch.isfinite(loss["loss_ce"]))
        self.assertEqual(float(loss["loss_dt"]), 0.0)


class GlobalLogRateTest(unittest.TestCase):
    """The learnable global log-rate decouples the event rate from vocab size.

    Without it the time loss uses rate = sum(exp(logit)) ~ vocab_size, which is
    orders of magnitude above the true rate and dominates cross-entropy. These
    tests pin the scalar's initialisation, trainability, optimiser handling, and
    the decoupling property that motivated it.
    """

    def _model(self, vocab_size, log_rate_init=None):
        config = FermatConfig(
            block_size=8,
            vocab_size=vocab_size,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            t_min=0.1,
            log_rate_init=log_rate_init,
        )
        return Fermat(config)

    def _time_loss_at_init(self, vocab_size):
        torch.manual_seed(0)
        model = self._model(vocab_size)
        idx = torch.tensor([[2, 3, 4, 5]])
        age = torch.tensor([[10.0, 25.0, 40.0, 70.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4, 5, 6]])
        targets_age = torch.tensor([[25.0, 40.0, 70.0, 120.0]])
        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            targets_age,
            target_token_type=token_type,
        )
        return model, loss

    def test_log_rate_initialises_to_negative_log_vocab(self):
        import math

        model = self._model(7699)
        self.assertAlmostEqual(float(model.log_rate), -math.log(7699), places=4)

    def test_explicit_log_rate_init_is_respected(self):
        model = self._model(64, log_rate_init=-3.0)
        self.assertAlmostEqual(float(model.log_rate), -3.0, places=5)

    def test_time_loss_does_not_scale_with_vocab_size(self):
        # The pathology was loss_dt growing ~linearly with vocab. With the
        # scalar both stay the same small order of magnitude despite a 64x gap.
        _, small = self._time_loss_at_init(64)
        _, large = self._time_loss_at_init(4096)
        self.assertTrue(torch.isfinite(small["loss_dt"]))
        self.assertTrue(torch.isfinite(large["loss_dt"]))
        self.assertLess(float(large["loss_dt"]), 50.0)
        self.assertLess(
            float(large["loss_dt"]),
            5.0 * float(small["loss_dt"]),
        )

    def test_log_rate_receives_gradient(self):
        model, loss = self._time_loss_at_init(64)
        (loss["loss_ce"] + loss["loss_dt"]).backward()
        self.assertIsNotNone(model.log_rate.grad)
        self.assertTrue(torch.isfinite(model.log_rate.grad))
        self.assertNotEqual(float(model.log_rate.grad), 0.0)

    def test_optimizer_groups_include_log_rate(self):
        model = self._model(64)
        optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        grouped = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        self.assertIn(id(model.log_rate), grouped)


class DecoupledTimeHeadTest(unittest.TestCase):
    """Option A: a separate head predicts the rate from the hidden state."""

    def _config(self, decoupled, vocab_size=64):
        return FermatConfig(
            block_size=8,
            vocab_size=vocab_size,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            t_min=0.1,
            decoupled_time_head=decoupled,
        )

    def _forward(self, model):
        torch.manual_seed(0)
        idx = torch.tensor([[2, 3, 4, 5]])
        age = torch.tensor([[10.0, 25.0, 40.0, 70.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4, 5, 6]])
        targets_age = torch.tensor([[25.0, 40.0, 70.0, 120.0]])
        return model(
            idx,
            age,
            token_type,
            targets,
            targets_age,
            target_token_type=token_type,
        )

    def test_decoupled_creates_time_head_not_log_rate(self):
        model = Fermat(self._config(True))
        self.assertTrue(hasattr(model, "time_head"))
        self.assertFalse(hasattr(model, "log_rate"))

    def test_coupled_creates_log_rate_not_time_head(self):
        model = Fermat(self._config(False))
        self.assertTrue(hasattr(model, "log_rate"))
        self.assertFalse(hasattr(model, "time_head"))

    def test_decoupled_exposes_finite_effective_log_rate(self):
        model = Fermat(self._config(True))
        _, loss, _ = self._forward(model)
        self.assertTrue(torch.isfinite(loss["loss_dt"]))
        effective = loss["effective_log_rate"]
        self.assertIsNotNone(effective)
        self.assertEqual(effective.shape, (1, 4))
        self.assertTrue(torch.isfinite(effective).all())

    def test_decoupled_time_head_receives_gradient(self):
        model = Fermat(self._config(True))
        _, loss, _ = self._forward(model)
        loss["loss_dt"].backward()
        self.assertIsNotNone(model.time_head.bias.grad)
        self.assertTrue(torch.isfinite(model.time_head.bias.grad).all())
        self.assertNotEqual(float(model.time_head.bias.grad.abs().sum()), 0.0)

    def test_decoupled_optimizer_groups_include_time_head(self):
        model = Fermat(self._config(True))
        optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        grouped = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        self.assertIn(id(model.time_head.weight), grouped)
        self.assertIn(id(model.time_head.bias), grouped)


class TwoStageTimeHeadTest(unittest.TestCase):
    def _model(self):
        return Fermat(FermatConfig(
            block_size=4,
            vocab_size=16,
            n_token_types=len(TokenType),
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
            t_min=0.1,
            decoupled_time_head=True,
            two_stage_time_head=True,
            mask_ties=True,
            ignore_types=[
                TokenType.PAD,
                TokenType.SEX,
                TokenType.NO_EVENT,
                TokenType.LAB,
            ],
        ))

    def test_same_day_and_different_day_losses_use_separate_targets(self):
        model = self._model()
        idx = torch.tensor([[2, 3, 4, 5]])
        age = torch.tensor([[10.0, 10.0, 20.0, 30.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4, 5, 6]])
        target_age = torch.tensor([[10.0, 20.0, 30.0, 30.0]])

        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            target_age,
            target_token_type=token_type,
        )

        self.assertEqual(int(loss["n_targets"]), 4)
        self.assertEqual(int(loss["n_dt_targets"]), 2)
        self.assertEqual(loss["same_day_logits"].shape, targets.shape)
        self.assertTrue(torch.isfinite(loss["loss_same_day"]))
        self.assertTrue(torch.isfinite(loss["loss_dt"]))

    def test_same_day_only_batch_has_finite_zero_different_day_loss(self):
        model = self._model()
        idx = torch.tensor([[2, 3]])
        age = torch.tensor([[10.0, 20.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4]])

        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            age.clone(),
            target_token_type=token_type,
        )

        self.assertEqual(int(loss["n_dt_targets"]), 0)
        self.assertTrue(torch.isfinite(loss["loss_dt"]))
        self.assertEqual(float(loss["loss_dt"].detach()), 0.0)

    def test_same_day_head_receives_gradient_and_is_optimized(self):
        model = self._model()
        idx = torch.tensor([[2, 3]])
        age = torch.tensor([[10.0, 20.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4]])
        target_age = torch.tensor([[10.0, 30.0]])
        _, loss, _ = model(
            idx,
            age,
            token_type,
            targets,
            target_age,
            target_token_type=token_type,
        )
        loss["loss_same_day"].backward()
        self.assertIsNotNone(model.same_day_head.bias.grad)
        self.assertTrue(torch.isfinite(model.same_day_head.bias.grad).all())

        optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        grouped = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        self.assertIn(id(model.same_day_head.weight), grouped)
        self.assertIn(id(model.same_day_head.bias), grouped)

    def test_same_day_logits_do_not_leak_the_target_age(self):
        # The tie mask hides current-date events only when the next event is
        # same-day, so the same-day head must read a target-independent causal
        # representation. Its logits must not change when targets_age changes.
        config = FermatConfig(
            block_size=8,
            vocab_size=64,
            n_token_types=len(TokenType),
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            t_min=0.1,
            mask_ties=True,
            decoupled_time_head=True,
            two_stage_time_head=True,
        )
        model = Fermat(config).eval()
        idx = torch.tensor([[2, 3, 4, 5]])
        age = torch.tensor([[10.0, 10.0, 10.0, 20.0]])
        token_type = torch.full_like(idx, TokenType.DX)
        targets = torch.tensor([[3, 4, 5, 6]])
        token_target_type = torch.full_like(targets, TokenType.DX)
        same_day_age = torch.tensor([[10.0, 10.0, 20.0, 40.0]])
        different_day_age = torch.tensor([[15.0, 25.0, 35.0, 45.0]])

        with torch.no_grad():
            _, loss_a, _ = model(
                idx, age, token_type, targets, same_day_age,
                target_token_type=token_target_type, return_attention=False,
            )
            _, loss_b, _ = model(
                idx, age, token_type, targets, different_day_age,
                target_token_type=token_target_type, return_attention=False,
            )

        torch.testing.assert_close(
            loss_a["same_day_logits"], loss_b["same_day_logits"]
        )


if __name__ == "__main__":
    unittest.main()
