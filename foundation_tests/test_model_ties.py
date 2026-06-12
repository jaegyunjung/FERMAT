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


if __name__ == "__main__":
    unittest.main()
