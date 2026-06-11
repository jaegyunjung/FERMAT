import unittest

import numpy as np
import torch

from model import TokenType
from utils import get_batch, get_p2i


def trajectory(length=20):
    return np.array(
        [
            [0, 100 + index, index + 1, TokenType.DX]
            for index in range(length)
        ],
        dtype=np.uint32,
    )


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.data = trajectory()
        self.p2i = get_p2i(self.data)

    def first_raw_token(self, selector):
        x, *_ = get_batch(
            torch.tensor([0]),
            self.data,
            self.p2i,
            select=selector,
            padding=None,
            block_size=4,
            no_event_token_rate=0,
        )
        return int(x[0, 0]) - 1

    def test_fixed_window_positions(self):
        self.assertEqual(self.first_raw_token("left"), 1)
        self.assertEqual(self.first_raw_token("middle"), 8)
        self.assertEqual(self.first_raw_token("right"), 16)

    def test_random_window_is_deterministic_for_same_batch(self):
        first = self.first_raw_token("random")
        second = self.first_raw_token("random")

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first, 16)

    def test_padding_none_is_supported(self):
        result = get_batch(
            torch.tensor([0]),
            self.data,
            self.p2i,
            select="left",
            padding=None,
            block_size=4,
            no_event_token_rate=5,
        )

        self.assertEqual(result[0].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
