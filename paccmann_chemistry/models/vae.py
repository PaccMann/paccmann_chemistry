"""Model Classes Module."""
from itertools import takewhile

import torch
import torch.nn as nn

from .stack_rnn import StackGRU
from ..utils.search import GreedySearch, BeamSearch, SamplingSearch


class StackGRUEncoder(StackGRU):
    """Stacked GRU Encoder."""

    def __init__(self, params):
        """
        Constructor.

        Args:
            params (dict): Hyperparameters.

        Items in params:
            latent_dim (int): Size of latent mean and variance.
            rnn_cell_size (int): Hidden size of GRU.
            vocab_size (int): Output size of GRU (vocab size).
            stack_width (int): Number of stacks in parallel.
            stack_depth (int): Stack depth.
            n_layers (int): The number of GRU layer.
            dropout (float): Dropout on the output of GRU layers except the
                last layer.
            batch_size (int): Batch size.
            lr (float, optional): Learning rate default 0.01.
            optimizer (str, optional): Choice from OPTIMIZER_FACTORY.
                Defaults to 'adadelta'.
            padding_index (int, optional): Index of the padding token.
                Defaults to 0.
            bidirectional (bool, optional): Whether to train a bidirectional
                GRU. Defaults to False.
        """
        super(StackGRUEncoder, self).__init__(params)

        self.bidirectional = params.get('bidirectional', False)
        self.n_directions = 2 if self.bidirectional else 1

        # In case of bidirectionality, we create a second StackGRU object that
        # will see the sequence in reverse direction
        if self.bidirectional:
            self.backward_stackgru = StackGRU(params)

        self.latent_dim = params['latent_dim']
        self.hidden_to_mu = nn.Linear(
            in_features=self.rnn_cell_size * self.n_directions,
            out_features=self.latent_dim
        )
        self.hidden_to_logvar = nn.Linear(
            in_features=self.rnn_cell_size * self.n_directions,
            out_features=self.latent_dim
        )

    def encoder_train_step(self, input_seq):
        """
        The Encoder Train Step.

        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
            of shape `[max batch sequence length +1, batch_size]`, where +1 is
            for the added start_index.

        Note: Input_seq is an output of sequential_data_preparation(batch) with
            batches returned by a DataLoader object.

        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar

            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is the log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """

        # Forward pass
        hidden = self.init_hidden()
        stack = self.init_stack()
        for input_entry in input_seq:
            output, hidden, stack = self(input_entry, hidden, stack)

        hidden = self._post_gru_reshape(hidden)

        # Backward pass:
        if self.bidirectional:
            assert len(input_seq.shape) == 2, 'Input Seq must be 2D Tensor.'
            hidden_backward = self.backward_stackgru.init_hidden()
            stack_backward = self.backward_stackgru.init_stack()

            # [::-1] not yet implemented in torch.
            # We roll up time from end to start
            for input_entry_idx in range(len(input_seq) - 1, -1, -1):
                output_backward, hidden_backward, stack_backward = (
                    self.backward_stackgru(
                        input_seq[input_entry_idx], hidden_backward,
                        stack_backward
                    )
                )
            # Concatenate forward and backward
            hidden_backward = self._post_gru_reshape(hidden_backward)
            hidden = torch.cat([hidden, hidden_backward], axis=1)

        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)

        return mu, logvar

    def _post_gru_reshape(self, hidden: torch.Tensor) -> torch.Tensor:
        expected_shape = torch.tensor(
            [self.n_layers, self.batch_size, self.rnn_cell_size]
        )
        if not torch.equal(torch.tensor(hidden.shape), expected_shape):
            raise ValueError(
                f'GRU hidden layer has incorrect shape: {hidden.shape}. '
                f'Expected shape: {expected_shape}'
            )

        # Layers x Batch x Cell_size ->  B x C
        hidden = hidden[-1, :, :]

        return hidden


class StackGRUDecoder(StackGRU):
    """Stack GRU Decoder."""

    def __init__(self, params, *args, **kwargs):
        """
        Constructor.

        Args:
            params (dict): Hyperparameters.

        Items in params:
            latent_dim (int): Size of latent mean and variance.
            rnn_cell_size (int): Hidden size of GRU.
            vocab_size (int): Output size of GRU (vocab size).
            stack_width (int): Number of stacks in parallel.
            stack_depth (int): Stack depth.
            n_layers (int): The number of GRU layer.
            dropout (float): Dropout on the output of GRU layers except the
                last layer.
            batch_size (int): Batch size.
            lr (float, optional): Learning rate default 0.01.
            optimizer (str, optional): Choice from OPTIMIZER_FACTORY.
                Defaults to 'adadelta'.
            padding_index (int, optional): Index of the padding token.
                Defaults to 0.
            bidirectional (bool, optional): Whether to train a bidirectional
                GRU. Defaults to False.
        """
        super(StackGRUDecoder, self).__init__(params, *args, **kwargs)
        self.params = params
        self.latent_dim = params['latent_dim']
        self.latent_to_hidden = nn.Linear(
            in_features=self.latent_dim, out_features=self.rnn_cell_size
        )

    def decoder_train_step(self, latent_z, input_seq, target_seq):
        """
        The Decoder Train Step.

        Args:
            latent_z (torch.Tensor): The sampled latent representation
                of the SMILES to be used for generation of shape
                `[1, batch_size, latent_dim]`.
            input_seq (torch.Tensor): The sequence of indices for the
                input of size `[max batch sequence length +1, batch_size]`,
                where +1 is for the added start_index.
            target_seq (torch.Tensor): The sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.

        Returns:
            The cross-entropy training loss for the decoder.
        """
        hidden = self.latent_to_hidden(latent_z)
        stack = self.init_stack()
        loss = 0
        for input_entry, target_entry in zip(input_seq, target_seq):
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden, stack
            )
            loss += self.criterion(output, target_entry.squeeze())

        return loss

    def generate_from_latent(
        self,
        latent_z,
        prime_input,
        end_token,
        search=SamplingSearch,
        generate_len=100
    ):
        """
        Generate SMILES From Latent Z.

        Args:
            latent_z (torch.Tensor): The sampled latent representation
                of size `[1, batch_size, latent_dim]`.
            prime_input (torch.Tensor): Tensor of indices for the priming
                string. Must be of size [prime_input length].
                Example: `prime_input = torch.tensor([2, 4, 5])`
            end_token (torch.Tensor): End token for the generated molecule
                of shape [1].
                Example: `end_token = torch.LongTensor([3])`
            search (paccmann_chemistry.utils.search.Search): search strategy
                used in the decoder.
            generate_len (int): Length of the generated molecule.

        Returns:
            torch.Tensor: The sequence(s) for the generated molecule(s)
                of shape [batch_size, generate_len + len(prime_input)].

        Note: For each generated sequence all indices after the first
            end_token must be discarded.
        """
        n_layers = self.n_layers
        latent_z = latent_z.repeat(n_layers, 1, 1)
        hidden = self.latent_to_hidden(latent_z)
        batch_size = hidden.shape[1]
        stack = self.init_stack(batch_size)
        generated_seq = prime_input.repeat(batch_size, 1)
        prime_input = generated_seq.transpose(1, 0)

        # use priming string to "build up" hidden state
        for prime_entry in prime_input[:-1]:
            _, hidden, stack = self(prime_entry, hidden, stack)
        input_token = prime_input[-1]

        # initialize beam search
        is_beam = isinstance(search, BeamSearch)
        if is_beam:
            beams = [[[list(), 0.0]]] * batch_size
            input_token = torch.stack(
                [input_token] +
                [input_token.clone() for _ in range(search.beam_width - 1)]
            )
            hidden = torch.stack(
                [hidden] +
                [hidden.clone() for _ in range(search.beam_width - 1)]
            )
            stack = torch.stack(
                [stack] +
                [stack.clone() for _ in range(search.beam_width - 1)]
            )

        for _ in range(generate_len):
            if not is_beam:
                output, hidden, stack = self(input_token, hidden, stack)
                top_idx = search.step(output.detach())
                # add generated_seq character to string and use as next input
                generated_seq = torch.cat((generated_seq, top_idx), dim=1)
                input_token = top_idx.view(1, -1)
                # if we don't generate in batches, we can do early stopping.
                if batch_size == 1 and top_idx == end_token:
                    break
            else:
                output, hidden, stack = zip(*[
                    self(an_input_token, a_hidden, a_stack)
                    for an_input_token, a_hidden, a_stack in zip(
                        input_token, hidden, stack
                    )
                ])  # yapf: disable
                output = torch.stack(output)
                hidden = torch.stack(hidden)
                stack = torch.stack(stack)
                input_token, beams = search.step(output.detach(), beams)
        if is_beam:
            generated_seq = torch.stack([
                # get the list of tokens with the highest score
                torch.tensor(beam[0][0]) for beam in beams
            ])  # yapf: disable
        return generated_seq


class TeacherVAE(nn.Module):

    def __init__(self, encoder, decoder):
        """
        Initialization.

        Args:
            encoder (StackGRUEncoder): the encoder object.
            decoder (StackGRUDecoder): the decoder object.
        """
        super(TeacherVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, input_seq):
        """
        VAE Encoder.

        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.

        Returns:
            mu (torch.Tensor): the latent mean of shape
                `[1, batch_size, latent_dim]`.
            logvar (torch.Tensor): log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        mu, logvar = self.encoder.encoder_train_step(input_seq)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample Z From Latent Dist.

        Args:
            mu (torch.Tensor): the latent mean of shape
                `[1, batch_size, latent_dim]`.
            logvar (torch.Tensor): log of the latent variance of shape
                `[1, batch_size, latent_dim]`.

        Returns:
            torch.Tensor: Sampled latent z from the latent distribution of
                shape `[1, batch_size, latent_dim]`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, latent_z, input_seq, target_seq):
        """
        Decode The Latent Z (for training).

        Args:
            latent_z (torch.Tensor): the sampled latent representation
                of the SMILES to be used for generation of shape
                `[1, batch_size, latent_dim]`
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
            target_seq (torch.Tensor): the sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.

        Returns:
            the cross-entropy training loss for the decoder.
        """
        n_layers = self.decoder.n_layers
        latent_z = latent_z.repeat(n_layers, 1, 1)
        decoder_loss = self.decoder.decoder_train_step(
            latent_z, input_seq, target_seq
        )
        return decoder_loss

    def forward(self, input_seq, decoder_seq, target_seq):
        """
        The Forward Function.

        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
            target_seq (torch.Tensor): the sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): decoder_loss, mu,
                logvar

            decoder_loss is the cross-entropy training loss for the decoder.
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        mu, logvar = self.encode(input_seq)
        latent_z = self.reparameterize(mu, logvar).unsqueeze(0)
        decoder_loss = self.decode(latent_z, decoder_seq, target_seq)
        return decoder_loss, mu, logvar

    def generate(
        self,
        latent_z,
        prime_input,
        end_token,
        generate_len=100,
        search=SamplingSearch
    ):
        """
        Generate SMILES From Latent Z.

        Args:
            latent_z (torch.Tensor): The sampled latent representation
                of size `[1, batch_size, latent_dim]`.
            prime_input (torch.Tensor): Tensor of indices for the priming
                string. Must be of size `[1, prime_input length]` or
                `[prime_input length]`.
                Example:
                    `prime_input = torch.tensor([2, 4, 5]).view(1, -1)`
                    or
                    `prime_input = torch.tensor([2, 4, 5])`
            end_token (torch.Tensor): End token for the generated molecule
                of shape `[1]`.
                Example: `end_token = torch.LongTensor([3])`
            generate_len (int): Length of the generated molecule.

        Returns:
            iterable: An iterator returning the torch tensor of
                sequence(s) for the generated molecule(s) of shape
                `[sequence length]`.

        Note: The start and end tokens are automatically stripped
            from the returned torch tensors for the generated molecule.
        """
        generated_batch = self.decoder.generate_from_latent(
            latent_z,
            prime_input,
            end_token,
            search=search,
            generate_len=generate_len
        )

        molecule_gen = (
            takewhile(lambda x: x != end_token, molecule[1:])
            for molecule in generated_batch
        )   # yapf: disable

        molecule_map = map(list, molecule_gen)
        molecule_iter = iter(map(torch.tensor, molecule_map))

        return molecule_iter

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)
