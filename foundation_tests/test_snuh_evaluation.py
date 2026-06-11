import unittest

import numpy as np
import torch

from model import TokenType
from scripts.evaluate_snuh_checkpoint import (
    clinical_output_mask,
    collate,
    iter_windows,
    window_start,
)


class SnuhEvaluationTest(unittest.TestCase):
    def test_window_start(self):
        self.assertEqual(window_start(100, 20, "left"), 0)
        self.assertEqual(window_start(100, 20, "middle"), 39)
        self.assertEqual(window_start(100, 20, "right"), 79)
        self.assertEqual(window_start(10, 20, "right"), 0)

    def test_repeated_targets_use_prior_dates(self):
        data = np.array(
            [
                [0, 10, 3, TokenType.DX],
                [0, 10, 3, TokenType.DX],
                [0, 20, 3, TokenType.DX],
                [0, 30, 4, TokenType.RX],
            ],
            dtype=np.uint32,
        )

        windows = list(iter_windows(data, 8, ["left"]))

        np.testing.assert_array_equal(
            windows[0]["repeated"],
            np.array([False, True, False]),
        )

    def test_collate_shifts_raw_token_ids(self):
        data = np.array(
            [
                [0, 10, 3, TokenType.DX],
                [0, 20, 4, TokenType.RX],
            ],
            dtype=np.uint32,
        )
        window = next(iter_windows(data, 8, ["left"]))

        x, _, y, _, xt, yt, repeated = collate([window], "cpu")

        self.assertEqual(int(x[0, 0]), 4)
        self.assertEqual(int(y[0, 0]), 5)
        self.assertEqual(int(xt[0, 0]), TokenType.DX)
        self.assertEqual(int(yt[0, 0]), TokenType.RX)
        self.assertFalse(bool(repeated[0, 0]))

    def test_clinical_output_mask_uses_shifted_ids(self):
        registry = [
            {"token_id": "0", "token_type_id": str(TokenType.PAD)},
            {"token_id": "1", "token_type_id": str(TokenType.DX)},
            {"token_id": "2", "token_type_id": str(TokenType.LAB)},
            {"token_id": "3", "token_type_id": str(TokenType.DTH)},
        ]

        mask = clinical_output_mask(registry, vocab_size=6, device="cpu")

        torch.testing.assert_close(
            mask,
            torch.tensor([False, False, True, False, True, False]),
        )


if __name__ == "__main__":
    unittest.main()
