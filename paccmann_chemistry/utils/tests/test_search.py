import unittest
import torch
from paccmann_chemistry.utils.search import (
    GreedySearch, SamplingSearch, BeamSearch
)

LOGITS = torch.tensor([
    [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]] * 5
]*2)


class TestGreedySearch(unittest.TestCase):
    """Testing the GreedySearch."""

    def test_forward(self) -> None:
        """Test greedy search."""
        search = GreedySearch()
        tokens = search(LOGITS)
        self.assertListEqual(
            tokens.numpy().tolist(), [[4, 0]*5]*2
        )


class TestSamplingSearch(unittest.TestCase):
    """Testing the SamplingSearch."""

    def test_forward(self) -> None:
        """Test sampling search."""
        search = SamplingSearch()
        tokens = search(LOGITS)
        self.assertListEqual(
            list(tokens.shape),
            list(LOGITS.shape[:2])
        )


class TestBeamSearch(unittest.TestCase):
    """Testing the BeamSearch."""

    def test_forward(self) -> None:
        """Test beam search."""
        search = BeamSearch()
        tokens, _ = search(LOGITS)
        self.assertListEqual(
            tokens.numpy().tolist(),
            [[
                [4, 4, 4],
                [0, 0, 0],
                [4, 4, 4],
                [0, 0, 0],
                [4, 4, 4],
                [0, 0, 0],
                [4, 4, 4],
                [0, 0, 0],
                [4, 4, 3],
                [0, 1, 0]
            ]]*2
        )
