import unittest
import numpy as np
import torch

from paccmann_chemistry.models.vae import StackGRUEncoder

# pylint: disable=not-callable, no-member


class TestStackGRUEncoder(unittest.TestCase):
    """Testing the StackGRUEncoder"""

    default_params = {
        'latent_dim': 128,
        'input_size': 12,
        'rnn_cell_size': 64,
        'embedding_size': 80,
        'output_size': 100,
        'stack_width': 50,
        'stack_depth': 51,
        'n_layers': 3,
        'dropout': 1,
        'batch_size': 32,
        'bidirectional': True
    }

    def test__post_gru_reshape(self) -> None:
        """Tests if the reshaping on the hidden layer of the GRU is correct.

        __Note: For more details look at issue #5__
        """
        params = self.default_params
        cell_size = params['rnn_cell_size']
        n_layers = params['n_layers']
        n_directions = 2 if params['bidirectional'] else 1
        batch_size = params['batch_size']

        gru_stack = StackGRUEncoder(params)

        correct_sample = np.arange(n_directions * cell_size)  # D*C

        # Emulate the hidden layer of the GRU
        hidden = np.tile(correct_sample, n_layers)  # Lx(D*C)
        hidden = hidden.reshape(
            n_layers * n_directions, cell_size
        )  # (L*D)xC
        hidden = [torch.Tensor(hidden) for _ in range(batch_size)]
        hidden = torch.stack(hidden, dim=1)  # LDxBxC

        hidden = gru_stack._post_gru_reshape(hidden)

        for sample in hidden.unbind():
            self.assertListEqual(sample.tolist(), correct_sample.tolist())

    def test__no_batch_mismatch__in__encoder_train_step(self) -> None:
        """Tests if there is any difference between equal samples in the same
        batch. If so it may indicate some crosstalk between samples in the 
        batch
        """
        params = self.default_params
        batch_size = params['batch_size']
        gru_stack = StackGRUEncoder(params)

        base_sample = np.arange(10)
        base = [torch.Tensor(base_sample) for _ in range(batch_size)]
        _input = torch.stack(base, dim=1).long()

        mus, logvars = gru_stack.encoder_train_step(_input)

        first_mu = mus[0].tolist()
        for _mu in mus.unbind():
            # NOTE: I assume there may be some tiny numerical differences
            # between the outputs (e.g. roundoff error)
            self.assertTrue(np.allclose(_mu.tolist(), first_mu))

        first_logvar = logvars[0].tolist()
        for _logvar in logvars.unbind():
            self.assertTrue(np.allclose(_logvar.tolist(), first_logvar))
