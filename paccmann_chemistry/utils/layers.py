"""Custom layers implementation."""
from collections import OrderedDict
from paccmann_predictor.utils.utils import Squeeze

import torch
import torch.nn as nn


class Permute21(nn.Module):
    """Switch with channels dim with the following one.
    That is N L E to N C=E L, where L is sequence length and E embedding
    dimensionality.
    """

    def forward(self, input):
        return input.permute(0, 2, 1)


def vectorize(
    in_channels,
    activation_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.,
):
    """Convolutional block reducing N, C, L to N, L.
    The reduction in channels is often referred to as 1x1 convolution (in 2D).
    It's followed by a squeeze.
    Args:
        in_channels (int): Number of input channels.
        activation_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    'convolve', conv_block(  # conv, act, do, bn
                        in_channels=in_channels,
                        out_channels=1,
                        kernel_size=1,
                        activation_fn=activation_fn,
                        batch_norm=batch_norm,
                        dropout=dropout,
                    )
                )
            ] + [
                ('squeeze', Squeeze()),
            ]
        )
    )


def conv_block(
    in_channels,
    out_channels,
    kernel_size,
    activation_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.,
):
    """Convolutional block with activation, dropout and optionally batch norm.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of convolution kernels.
        kernel_size (tuple[int, int]): Size of the convolution kernels.
        activation_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    'convolve',
                    torch.nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2  # pad for valid convolution
                    )
                ),
                ('activation_fn', activation_fn),
                ('dropout', nn.Dropout(p=dropout)),
                (
                    'batch_norm', nn.BatchNorm1d(out_channels)
                    if batch_norm else nn.Identity()
                )
            ]
        )
    )


class TimeDistributed(nn.Module):
    """
    From:
    https://github.com/cxhernandez/molencoder/blob/master/molencoder/utils.py
    """

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y