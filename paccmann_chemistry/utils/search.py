"""Decoding utilities."""
import torch
from torch import nn


class Search(nn.Module):
    """Base search class."""

    def forward(self, logits: torch.Tensor) -> object:
        """
        Perform the search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            object: the search output.
        """
        raise NotImplementedError


class GreedySearch(Search):
    """"Greedy search."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform the greedy search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        return torch.argmax(logits, 2)


class SamplingSearch(Search):
    """"Sampling search."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform the sampling search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        probabilities = torch.softmax(logits, 2)
        return torch.stack([
            torch.multinomial(probability, 1)
            for probability in probabilities
        ]).squeeze()


class BeamSearch(Search):
    """Beam search."""

    def __init__(self, top_k=3):
        """
        Initialize the beam search.

        Args:
            top_k (int, optional): top sequences returned. Defaults to 3.
        """
        super().__init__()
        self.top_k = top_k

    def _beam_per_sequence(self, logits: torch.Tensor):
        """
        Beam per sequence in the batch.

        Args:
            logits (torch.Tensor): logits.
                (length, vocabulary_size)

        Returns:
            tuple: a tuple containing:
            - a tensor with tokens. (length, top_k)
            - score socre. (top_k)
        """
        beams = [[list(), 0.0]]
        probabilities = torch.softmax(logits, 1)
        from math import log
        from sys import float_info
        # walk over each step in sequence
        for probability in probabilities.numpy():
            all_candidates = list()
            # expand each current candidate
            for i in range(len(beams)):
                a_sequence, score = beams[i]
                for j in range(len(probability)):
                    candidate = [
                        a_sequence + [j],
                        score + log(probability[j] + float_info.epsilon)
                    ]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(
                all_candidates, key=lambda pair: pair[1], reverse=True
            )
            # select k best
            beams = ordered[:self.top_k]
        sequences, scores = zip(*beams)
        return (torch.tensor(list(sequences)).T, torch.tensor(list(scores)))

    def forward(self, logits: torch.Tensor) -> tuple:
        """
        Perform the bean search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            tuple: a tuple containing:
            - the token indexes for each top sequence.
                (batch_size, length, top_k)
            - the scores. (batch_size, top_k)
        """
        tokens, scores = zip(*[
            self._beam_per_sequence(sequence)
            for sequence in logits
        ])
        return (
            torch.stack(tokens),
            torch.stack(scores)
        )
