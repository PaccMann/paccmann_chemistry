import unittest
import numpy as np
import torch

from paccmann_chemistry.models.vae import StackGRUEncoder

# pylint: disable=not-callable, no-member


class TestStackGRUEncoder(unittest.TestCase):
    """Testing the StackGRUEncoder"""

    default_params = {
        'latent_dim': 128,
        'rnn_cell_size': 30,
        'embedding_size': 80,
        'vocab_size': 100,
        'stack_width': 50,
        'stack_depth': 51,
        'n_layers': 3,
        'dropout': .7,
        'batch_size': 32,
        'bidirectional': True
    }

    def assertListsClose(self, var1, var2, rtol=1e-5, atol=1e-7):
        self.assertTrue(np.allclose(var1, var2, rtol=rtol, atol=atol))

    def test__post_gru_reshape(self) -> None:
        """Tests if the reshaping on the hidden layer of the GRU is correct.

        __Note: For more details look at issue #5__
        """
        params = self.default_params
        cell_size = params['rnn_cell_size']
        n_layers = params['n_layers']
        batch_size = params['batch_size']

        gru_stack = StackGRUEncoder(params)

        correct_sample = np.arange(cell_size)  # C

        # Emulate the hidden layer of the GRU
        hidden = np.tile(correct_sample, n_layers)  # LxC
        hidden = hidden.reshape(n_layers, cell_size)  # LxC
        hidden = [torch.Tensor(hidden) for _ in range(batch_size)]
        hidden = torch.stack(hidden, dim=1)  # LxBxC

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
        gru_stack.eval()

        base_sample = np.arange(10)
        base = [torch.Tensor(base_sample) for _ in range(batch_size)]
        _input = torch.stack(base, dim=1).long()

        mus, logvars = gru_stack.encoder_train_step(_input)

        first_mu = mus[0].tolist()
        for _mu in mus.unbind():
            # NOTE: I assume there may be some tiny numerical differences
            # between the outputs (e.g. roundoff error)
            self.assertTrue(
                np.allclose(_mu.tolist(), first_mu, rtol=1e-5, atol=1e-7)
            )

        first_logvar = logvars[0].tolist()
        for _logvar in logvars.unbind():
            self.assertListsClose(_logvar.tolist(), first_logvar)

    def test__encoding_independent_from_batch_with_stack(self) -> None:
        self._encoding_independent_from_batch(use_stack=True)

    def test__encoding_independent_from_batch_no_stack(self) -> None:
        self._encoding_independent_from_batch(use_stack=False)

    def _encoding_independent_from_batch(self, use_stack) -> None:
        """Test that the results of a model are gonna be consistent
        regadless of the model's batch size"""

        params = self.default_params
        params['batch_size'] = 128
        params['vocab_size'] = 55
        params['use_stack'] = use_stack

        device = torch.device('cpu')

        gru_encoder = StackGRUEncoder(params).to(device)
        state_dict = gru_encoder.state_dict()

        # 2 (start index) + ordered token sequence
        sample = np.concatenate([[2], np.arange(3, 53)])

        def _get_sample_at_batch_size(batch_size):
            """Helper function to iterate over the batches"""
            params['batch_size'] = batch_size
            gru_encoder = StackGRUEncoder(params).to(device)
            gru_encoder.load_state_dict(state_dict)
            gru_encoder = gru_encoder.eval()

            # Setup a batch to be passed into the encoder. This is the same
            # that `training.sequential_data_preparation` does
            batch = np.stack(
                [sample for _ in range(params['batch_size'])], axis=1
            )
            encoder_seq = torch.tensor(batch).long()
            return gru_encoder.encoder_train_step(encoder_seq)[0][0]

        batch_sizes = [1, 2, 4, 12, 55, 128]

        results_by_batches = [
            _get_sample_at_batch_size(b) for b in batch_sizes
        ]

        for i, res1 in enumerate(results_by_batches):
            for j, res2 in enumerate(results_by_batches[i + 1:]):
                self.assertListsClose(res1.tolist(), res2.tolist())
