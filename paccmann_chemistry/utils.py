"""Utilities functions."""
import logging
from time import perf_counter
from functools import wraps
import math
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import copy

logger = logging.getLogger(__name__)


def seq_data_prep(
    input_batch, input_keep=1, start_index=2, end_index=3, dropout_index=1
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (torch.Tensor): batch of padded sequences, output of
        nn.utils.rnn.pad_sequence(batch) of size
            [sequence length, batch_size, 1].
        input_keep (float): the keep probability of input sequence tokens
            will keep or drop input tokens according to a Bernoulli distribution
            with p = input_keep. Defaults to 1.
        start_index (int): the index of the sequence start token
        end_index (int): the index of the sequence end token
        dropout_index (int): the index of the dropout token, defaults
            to '<UNK>'.

    Returns:
        encoder_seq (torch.Tensor): Batch of padded input sequences starting
            with the start_index, of size [sequence length +1, batch_size, 1].
        decoder_seq (torch.Tensor): Same like encoder_seq but word dropout is
            applied (it input_keep=1 -> encoder_seq = decoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
            in the end_index, of size [sequence length +1, batch_size, 1].
    """
    batch_size = input_batch.shape[1]
    input_batch = torch.LongTensor(input_batch.cpu().numpy())
    decoder_batch = input_batch.clone()
    # apply token dropout if keep != 1
    if input_keep != 1:
        # build dropout indices consisting of dropout_index
        dropout_indices = torch.LongTensor(
            dropout_index * torch.ones(1, batch_size).numpy()
        ).unsqueeze(2)
        # mask for token dropout
        mask = Bernoulli(input_keep).sample((input_batch.shape[0], ))
        mask = torch.LongTensor(mask.numpy())
        dropout_loc = np.where(mask == 0)[0]

        decoder_batch[dropout_loc] = dropout_indices

    start_indices = torch.LongTensor(
        start_index * torch.ones(1, batch_size).numpy()
    ).unsqueeze(2)
    input_seq = torch.cat((start_indices, input_batch), dim=0)
    decoder_seq = torch.cat((start_indices, decoder_batch), dim=0)

    def add_endidx(seq, end_index=end_index):
        end_idx = None
        if len(np.where(seq == 0)[0]) != 0:
            end_idx = np.where(seq == 0)[0][0]
        if end_idx:
            seq[end_idx] = end_index

    end_padding = torch.LongTensor(torch.zeros(1, batch_size).numpy()
                                   ).unsqueeze(2)
    target_seq = torch.cat((input_batch, end_padding), dim=0)
    target_seq = copy.deepcopy(target_seq).transpose(1, 0)
    target_seq = target_seq.transpose(1, 0)
    device = get_device()
    return input_seq.to(device), decoder_seq.to(device), target_seq.to(device)


def collate_fn(batch):
    """
    Collate funciton for DataLoader.

    Note: to be used as collate_fn in
    torch.utils.data.DataLoader.

    Args:
        batch: batch of sequences.

    Returns:
        sorted batch from longest to shortest.
    """
    return [
        batch[index] for index in map(
            lambda t: t[0],
            sorted(
                enumerate(batch), key=lambda t: t[1].shape[0], reverse=True
            )
        )
    ]


def track_loss(fn):
    """Decorator for tracking training loss."""
    lst = []

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal lst
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        elapsed = end - start
        logger.info(f'One epoch took {int(elapsed/60)} minutes.')
        lst += result
        return lst

    return inner


def kl_weight(step, growth_rate=0.004):
    """
    KL divergence weighting function for better training of
    encoder and decoder of the VAE.
    See: https://arxiv.org/abs/1511.06349

    Args:
        step (int): the training step.
        growth_rate (float): the rate at which the
            weight grows. Defaults to 0.0015 resulting
            in a weight of 1 around step=9000.

    Returns:
        weight (float): the weight of KL divergence
            loss term.
    """
    weight = 1 / (1 + math.exp((15 - growth_rate * step)))
    return weight


def get_device():
    return torch.device("cuda" if cuda() else "cpu")


def cuda():
    return torch.cuda.is_available()


def to_np(x):
    return x.data.cpu().numpy()


class Identity(nn.Module):
    """Wrapper for Identity activation function."""

    def forward(self, data):
        return data
