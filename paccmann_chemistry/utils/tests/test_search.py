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
            tokens.numpy().tolist(), [[4, 0]*LOGITS.shape[2]]*LOGITS.shape[0]
        )

    def test_step(self) -> None:
        """Test step-wise sampling search."""
        search = GreedySearch()
        groundtruth_tokens = [4, 0]*LOGITS.shape[2]
        for logits, token in zip(LOGITS.permute(1, 0, 2), groundtruth_tokens):
            tokens = search.step(logits)
            self.assertListEqual(
                list(tokens),
                list([token]*LOGITS.shape[0])
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

    def test_step(self) -> None:
        """Test step-wise sampling search."""
        search = SamplingSearch()
        for logits in LOGITS.permute(1, 0, 2):
            tokens = search.step(logits)
            self.assertListEqual(
                list(tokens.shape),
                list([LOGITS.shape[0], 1])
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
            ]]*LOGITS.shape[0]
        )

    def test_step(self) -> None:
        """Test step-wise beam search."""
        search = BeamSearch()
        # intialize beams for all the elements in the batch
        beams = [[[list(), 0.0]]]*LOGITS.shape[0]
        for logits in LOGITS.permute(1, 0, 2):
            tokens, beams = search.step(logits, beams)
        self.assertListEqual(
            [
                list(map(
                    list,
                    zip(*[beam[0] for beam in sample_beams])
                ))
                for sample_beams in beams
            ],
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
            ]]*LOGITS.shape[0]
        )
