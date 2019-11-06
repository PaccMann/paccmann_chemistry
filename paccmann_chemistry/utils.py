"""Utilities functions."""
import copy
import logging
import math
from functools import wraps
from time import perf_counter

import numpy as np
import tensorflow as tf
import torch
from torch.distributions.bernoulli import Bernoulli

logger = logging.getLogger(__name__)


def sequential_data_preparation(
    input_batch, input_keep=1, start_index=2, end_index=3, dropout_index=1
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The keep probability of input sequence tokens
            will keep or drop input tokens according to a Bernoulli
            distribution with p = input_keep. Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor): encoder_seq, decoder_seq,
        target_seq

        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size, 1]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size, 1]`.
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

    end_padding = torch.LongTensor(torch.zeros(1, batch_size).numpy()
                                   ).unsqueeze(2)
    target_seq = torch.cat((input_batch, end_padding), dim=0)
    target_seq = copy.deepcopy(target_seq).transpose(1, 0)
    target_seq = target_seq.transpose(1, 0)
    device = get_device()
    return input_seq.to(device), decoder_seq.to(device), target_seq.to(device)


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Note: to be used as collate_fn in torch.utils.data.DataLoader.

    Args:
        batch: Batch of sequences.

    Returns:
        Sorted batch from longest to shortest.
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


class TBLogger:
    """Tensorboard Logger."""

    def __init__(self, logdir):
        """
        Tensorboard Logger.

        Args:
            logdir (str): Directory of the event file to be written.
        """
        self.writer = tf.compat.v1.summary.FileWriter(logdir)

    def close(self):
        self.writer.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
        )
        self.writer.add_summary(summary, step)

    def log_scalar(self, scalar_tag, value, global_step):
        """
        Logs Scalar to Event File.

        Args:
            scalar_tag (str): The name of the scalar to be logged.
            value (torch.Tensor or np.array): The values to be logged.
            global_step (int): The logging step.

        Example:
            `tensorboard = TBLogger(log_dir)
            x = np.arange(1,101)
            y = 20 + 3 * x + np.random.random(100) * 100
            for i in range(0,100):
                tensorboard.log_scalar('myvalue', y[i], i)`
        """
        summary = tf.Summary()
        summary.value.add(tag=scalar_tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()


def kl_weight(step, growth_rate=0.004):
    """Kullback-Leibler weighting function.

    KL divergence weighting for better training of
    encoder and decoder of the VAE.

    Reference:
        https://arxiv.org/abs/1511.06349

    Args:
        step (int): The training step.
        growth_rate (float): The rate at which the weight grows.
            Defaults to 0.0015 resulting in a weight of 1 around step=9000.

    Returns:
        float: The weight of KL divergence loss term.
    """
    weight = 1 / (1 + math.exp((15 - growth_rate * step)))
    return weight


def get_device():
    return torch.device("cuda" if cuda() else "cpu")


def cuda():
    return torch.cuda.is_available()


def to_np(x):
    return x.data.cpu().numpy()
