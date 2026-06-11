import unittest

import torch

from model import align_time_deltas, build_attention_mask


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


if __name__ == "__main__":
    unittest.main()
