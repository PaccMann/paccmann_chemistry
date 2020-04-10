"""Decoding utilities."""
import torch
from torch import nn
from math import log
from sys import float_info


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

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform a greedy search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes for all the batch. (batch_size).
        """
        return torch.argmax(logits, 1)


class SamplingSearch(Search):
    """"Sampling search."""

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the sampling search.

        Args:
            temperature (float, optional): temperature parameter. Defaults to
                1.0, a.k.a., no temperature.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform the sampling search.

        Args:
            logits: torch.Tensor (Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes selected. (batch_size, length)
        """
        probabilities = torch.softmax(logits.div(self.temperature), 2)
        return torch.stack([
            torch.multinomial(probability, 1)
            for probability in probabilities
        ]).squeeze()

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform a sampling search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
        Returns:
            torch.Tensor: the token indexes for all the batch. (batch_size).
        """
        probabilities = torch.softmax(logits.div(self.temperature), 1)
        return torch.stack([
            torch.multinomial(probability, 1)
            for probability in probabilities
        ])


class BeamSearch(Search):
    """Beam search."""

    def __init__(self, top_k: int = 3, temperature: float = 1.0):
        """
        Initialize the beam search.

        Args:
            top_k (int, optional): top sequences returned. Defaults to 3.
            temperature (float, optional): temperature parameter. Defaults to
                1.0, a.k.a., no temperature.
        """
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def _beam_step_per_sequence(
        self, probability: torch.Tensor, beams: list
    ) -> list:
        """
        Perform a beam search step.

        Args:
            probability (torch.Tensor): probability for the current step.
                (vocabulary_size).
            beams (list): beams containg sequence and score.

        Returns:
            list: updated beams.
        """
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
        return ordered[:self.top_k]

    def _beam_per_sequence(self, logits: torch.Tensor) -> tuple:
        """
        Beam per sequence in the batch.

        Args:
            logits (torch.Tensor): logits.
                (length, vocabulary_size)

        Returns:
            tuple: a tuple containing:
            - a tensor with tokens. (length, top_k)
            - score. (top_k)
        """
        beams = [[list(), 0.0]]
        probabilities = torch.softmax(logits.div(self.temperature), 1)
        # walk over each step in sequence
        for probability in probabilities:
            beams = self._beam_step_per_sequence(probability, beams)
        sequences, scores = zip(*beams)
        return (torch.tensor(list(sequences)).T, torch.tensor(list(scores)))

    def forward(self, logits: torch.Tensor) -> tuple:
        """
        Perform the beam search.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, length, vocabulary_size)
        Returns:
            tuple: a tuple containing:
            - the token indexes for each top sequence.
                (batch_size, length, top_k)
            - scores. (batch_size, top_k)
        """
        tokens, scores = zip(*[
            self._beam_per_sequence(sequence)
            for sequence in logits
        ])
        return (
            torch.stack(tokens),
            torch.stack(scores)
        )

    def step(self, logits: torch.Tensor, beams: list) -> tuple:
        """
        Perform a beam search step.

        Args:
            logits (torch.Tensor): the model's
                logits. (batch_size, vocabulary_size)
            beams (list): beams for all the batch.
        Returns:
            tuple: a tuple containing:
            - the token indexes for all the batch.
                (batch_size, top_k)
            - updated beams for all the batch.
        """
        probabilities = torch.softmax(logits.div(self.temperature), 1)
        updated_beams = [
            self._beam_step_per_sequence(sample_probability, sample_beams)
            for sample_probability, sample_beams in zip(probabilities, beams)
        ]
        token_beams = torch.stack([
            # get last token for each beam
            torch.tensor([beam[0][-1] for beam in sample_beams])
            for sample_beams in updated_beams
        ])
        return (
            token_beams,
            updated_beams
        )
