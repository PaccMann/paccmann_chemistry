"""
SMILES Encoder following Gomez-Bombarelli landmark paper.
https://pubs.acs.org/doi/pdf/10.1021/acscentsci.7b00572
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from paccmann_omics.encoders.encoder import Encoder
from paccmann_omics.utils.hyperparams import ACTIVATION_FN_FACTORY
from paccmann_predictor.utils.utils import get_device
from paccmann_predictor.utils.layers import dense_layer
from paccmann_chemistry.utils.layers import conv_block, Permute21


class ConvEncoder(Encoder):
    """SMILES Encoder following Gomez-Bombarelli landmark paper. """

    def __init__(self, params, *args, **kwargs):
        """Constructor.
        Args:
            params (dict): A dictionary containing the parameter to built the
                convolutional Decoder.
                TODO params should become actual arguments (use **params).
        Required items in params:
            padding_length (int): size of sequence.
            embedding_size (int): size of the embedding.
            vocabulary_size (int): size amino acid vocabulary.
            latent_size (int): size of latent mean and variance.
        Items in params:
            filters (list[int], optional): numbers of filters to learn per
                convolutional layer. Defaults to [16, 16, 16, 16, 16].
            kernel_sizes (list[int], optional): sizes of kernel widths along
                along sequence per convolutional layer.
                Defaults to [3, 5, 7, 9, 11].
            activation_fn (string, optional): activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Default 'relu'.
            batch_norm (bool, optional): whether batch normalization is
                applied. Default True.
            dropout (float, optional): dropout probability in all
                except parametric layer. Default 0.0.
            embed_scale_grad (bool):  # TODO for embedding
            *args, **kwargs: positional and keyword arguments are ignored.
        """
        super(ConvEncoder, self).__init__(*args, **kwargs)
        self.device = get_device()

        self.padding_length = params.get('padding_length', 120)
        self.filters = params.get('filters', [9, 9, 11])
        self.embedding_size = params['embedding_size']
        self.vocabulary_size = params['vocabulary_size']

        self.kernel_sizes = params.get('kernel_sizes', [9, 9, 11])
        self.latent_size = params['latent_size']
        self.dense_size = params.get('encoder_dense_size', 196)

        self._sanity_checks()

        self.activation_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'tanh')]
        self.dropout = params.get('dropout', 0.0)
        self.batch_norm = params.get('batch_norm', True)

        # Embeddings
        self.embedding = nn.Embedding(
            self.vocabulary_size,
            self.embedding_size,
            scale_grad_by_freq=params.get('embed_scale_grad', False)
        )

        # Stack convolutional layers
        # output is of size N, L
        self.encoding = nn.Sequential(
            OrderedDict(
                [
                    ('permute', Permute21())  # embedding dim as channels
                ] + [
                    (
                        f'conv_block_{index}',
                        conv_block(  # conv, act, do, bn
                            in_channels=(
                                self.filters[index-1]
                                if index != 0 else self.embedding_size
                            ),
                            out_channels=num_kernel,
                            kernel_size=kernel_size,
                            activation_fn=self.activation_fn,
                            batch_norm=self.batch_norm,
                            dropout=self.dropout,
                        ).to(self.device)
                    ) for index, (num_kernel, kernel_size) in
                    enumerate(zip(self.filters, self.kernel_sizes))
                ]
            )
        )
        self.dense_layer = dense_layer(
            self.filters[-1] * self.padding_length,
            params.get('dense_size', 196),
            act_fn=self.activation_fn,
            dropout=self.dropout
        )

        # transform into VAE latent space
        self.encoding_to_mu = nn.Linear(self.dense_size, self.latent_size)
        self.encoding_to_logvar = nn.Linear(self.dense_size, self.latent_size)

    def encoder_train_step(self, x):
        """Alias for forward function """
        return self(x)

    def forward(self, x):
        """Projects an input into the latent space.
        Args:
            x (torch.Tensor): of type int and shape
                `[bs, padding_length]`.
        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar
            mu (torch.Tensor): Latent means of shape
                `[batch_size, self.latent_size]`.
            logvar (torch.Tensor): Latent log-std of shape
                `[batch_size, self.latent_size]`.
        """

        batch_size = x.shape[0]
        embedded = self.embedding(x.to(torch.int64))
        convolved = self.encoding(embedded).view(batch_size, -1)
        dense = self.dense_layer(convolved)
        mus = self.encoding_to_mu(dense)
        logvars = self.encoding_to_logvar(dense)

        return mus, logvars

    def _sanity_checks(self):
        """Checks size issues."""
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                f'Number of filters ({len(self.filters)}) does not match '
                f'number of kernel_sizes ({len(self.kernel_sizes)})'
            )
        assert self.kernel_sizes[-1] % 2 != 0, 'Kernel size should be odd.'


class GentrlGRUEncoder(nn.Module):

    def __init__(
        self,
        params,
        latent_size=50,
        bidirectional=False,
        pad_size=50,
        *args,
        **kwargs
    ):
        super(GentrlGRUEncoder, self).__init__()

        self.bidirectional = params.get('bidirectional', False)
        self.latent_size = params['latent_size']

        self.rnn_cell_size = params.get('rnn_cell_size', 256)
        self.n_layers = params['n_layers']
        self.vocabulary_size = params['vocabulary_size']

        self.pad_size = pad_size

        self.embs = nn.Embedding(self.vocabulary_size, self.rnn_cell_size)
        self.rnn = nn.GRU(
            input_size=self.rnn_cell_size,
            hidden_size=self.rnn_cell_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.rnn_cell_size, self.rnn_cell_size), nn.LeakyReLU(),
            nn.Linear(self.rnn_cell_size, 2 * self.latent_size)
        )

    def encoder_train_step(self, x):
        """Alias for forward function """
        return self(x)

    def forward(self, x):
        """Projects an input into the latent space.
        Args:
            x (torch.Tensor): of type int and shape
                `[bs, padding_length]`.
        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar
            mu (torch.Tensor): Latent means of shape
                `[batch_size, self.latent_size]`.
            logvar (torch.Tensor): Latent log-std of shape
                `[batch_size, self.latent_size]`.
        """

        outputs, _ = self.rnn(self.embs(x))
        code = self.final_mlp(outputs)

        mus, logvars = torch.split(code, self.latent_size, dim=1)
        return mus, logvars
