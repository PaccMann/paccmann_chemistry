"""Model Classes Module."""
from itertools import takewhile

import torch
import torch.nn as nn

from .stack_rnn import StackGRU
from ..utils.search import BeamSearch, SamplingSearch
from ..utils.hyperparams import OPTIMIZER_FACTORY

from .. import utils


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
        hidden = self.init_hidden
        stack = self.init_stack

        hidden = self._forward_fn(input_seq, hidden, stack)

        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)

        return mu, logvar

    def _forward_pass_padded(self, input_seq, hidden, stack):
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
        if isinstance(input_seq, nn.utils.rnn.PackedSequence) or \
                not isinstance(input_seq, torch.Tensor):
            raise TypeError('Input is PackedSequence or is not a Tensor')
        expanded_input_seq = input_seq.unsqueeze(1)
        for input_entry in expanded_input_seq:
            _output, hidden, stack = self(input_entry, hidden, stack)

        hidden = self._post_gru_reshape(hidden)

        # Backward pass:
        if self.bidirectional:
            assert len(input_seq.shape) == 2, 'Input Seq must be 2D Tensor.'
            hidden_backward = self.backward_stackgru.init_hidden
            stack_backward = self.backward_stackgru.init_stack

            # [::-1] not yet implemented in torch.
            # We roll up time from end to start
            for input_entry_idx in range(len(expanded_input_seq) - 1, -1, -1):

                _output_backward, hidden_backward, stack_backward = (
                    self.backward_stackgru(
                        expanded_input_seq[input_entry_idx], hidden_backward,
                        stack_backward
                    )
                )
            # Concatenate forward and backward
            hidden_backward = self._post_gru_reshape(hidden_backward)
            hidden = torch.cat([hidden, hidden_backward], axis=1)
        return hidden

    def _forward_pass_packed(self, input_seq, hidden, stack):
        """
        The Encoder Train Step.

        Args:
            input_seq (torch.nn.utls.rnn.PackedSequence): the sequence of
            indices for the input of shape.

        Note: Input_seq is an output of sequential_data_preparation(batch) with
            batches returned by a DataLoader object.

        Returns:
            (torch.Tensor, torch.Tensor): mu, logvar

            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is the log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        if not isinstance(input_seq, nn.utils.rnn.PackedSequence):
            raise TypeError('Input is not PackedSequence')

        final_hidden = hidden.detach().clone()
        final_stack = stack.detach().clone()
        input_seq_packed, batch_sizes = utils.perpare_packed_input(input_seq)

        prev_batch = batch_sizes[0]

        for input_entry, batch_size in zip(input_seq_packed, batch_sizes):
            final_hidden, hidden = utils.manage_step_packed_vars(
                final_hidden, hidden, batch_size, prev_batch, batch_dim=1
            )
            final_stack, stack = utils.manage_step_packed_vars(
                final_stack, stack, batch_size, prev_batch, batch_dim=0
            )
            prev_batch = batch_size
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden, stack
            )

        left_dims = hidden.shape[1]
        final_hidden[:, :left_dims, :] = hidden[:, :left_dims, :]
        final_stack[:left_dims, :, :] = stack[:left_dims, :, :]

        hidden = final_hidden
        stack = final_stack
        hidden = self._post_gru_reshape(hidden)

        # Backward pass:
        if self.bidirectional:
            # assert len(input_seq.shape) == 2, 'Input Seq must be 2D Tensor.'
            hidden_backward = self.backward_stackgru.init_hidden
            stack_backward = self.backward_stackgru.init_stack

            input_seq = utils.unpack_sequence(input_seq)

            for i, seq in enumerate(input_seq):
                idx = [i for i in range(len(seq) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                input_seq[i] = seq.index_select(0, idx)

            input_seq = utils.repack_sequence(input_seq)
            input_seq_packed, batch_sizes = utils.perpare_packed_input(
                input_seq
            )

            final_hidden = hidden_backward.detach().clone()
            prev_batch = batch_sizes[0]

            for input_entry, batch_size in zip(input_seq_packed, batch_sizes):
                # for seq in input_seq:
                final_hidden, hidden_backward = utils.manage_step_packed_vars(
                    final_hidden,
                    hidden_backward,
                    batch_size,
                    prev_batch,
                    batch_dim=1
                )
                final_stack, stack_backward = utils.manage_step_packed_vars(
                    final_stack,
                    stack_backward,
                    batch_size,
                    prev_batch,
                    batch_dim=0
                )
                prev_batch = batch_size
                output_backward, hidden_backward, stack_backward = (
                    self.backward_stackgru(
                        input_entry.unsqueeze(0), hidden_backward,
                        stack_backward
                    )
                )
            left_dims = hidden_backward.shape[1]
            final_hidden[:, :left_dims, :] = hidden_backward[:, :left_dims, :]
            hidden_backward = final_hidden

            # Concatenate forward and backward
            hidden_backward = self._post_gru_reshape(hidden_backward)
            hidden = torch.cat([hidden, hidden_backward], axis=1)
        return hidden

    def _post_gru_reshape(self, hidden: torch.Tensor) -> torch.Tensor:

        if not torch.equal(torch.tensor(hidden.shape), self.expected_shape):
            raise ValueError(
                f'GRU hidden layer has incorrect shape: {hidden.shape}. '
                f'Expected shape: {self.expected_shape}'
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
        self.output_layer = nn.Linear(self.rnn_cell_size, self.vocab_size)

        # 0 is padding index.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OPTIMIZER_FACTORY[
            params.get('optimizer', 'adadelta')
        ](self.parameters(), lr=params.get('lr', 0.01))  # yapf: disable

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
        stack = self.init_stack

        loss = self._forward_fn(input_seq, target_seq, hidden, stack)
        return loss

    def _forward_pass_padded(self, input_seq, target_seq, hidden, stack):
        """The Decoder Train Step.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by
            a DataLoader object.
        Returns:
            The cross-entropy training loss for the decoder.
        """
        if isinstance(input_seq, nn.utils.rnn.PackedSequence) and \
                not isinstance(input_seq, torch.Tensor):
            raise TypeError('Input is PackedSequence or is not a Tensor')

        loss = 0
        outputs = []
        for idx, (input_entry,
                  target_entry) in enumerate(zip(input_seq, target_seq)):
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden, stack
            )
            output = self.output_layer(output).squeeze()
            loss += self.criterion(output, target_entry.squeeze())
            outputs.append(output)

        # For monitoring purposes
        outputs = torch.stack(outputs, -1)
        self.outputs = torch.argmax(outputs, 1)

        return loss

    def _forward_pass_packed(self, input_seq, target_seq, hidden, stack):
        """The Decoder Train Step.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.
        Returns:
            The cross-entropy training loss for the decoder.
        """
        if not isinstance(input_seq, nn.utils.rnn.PackedSequence):
            raise TypeError('Input is not PackedSequence')

        loss = 0
        input_seq_packed, batch_sizes = utils.perpare_packed_input(input_seq)
        # Target sequence should have same batch_sizes as input_seq
        target_seq_packed, _ = utils.perpare_packed_input(target_seq)
        prev_batch = batch_sizes[0]
        outputs = []
        for idx, (input_entry, target_entry, batch_size) in enumerate(
            zip(input_seq_packed, target_seq_packed, batch_sizes)
        ):
            _, hidden = utils.manage_step_packed_vars(
                None, hidden, batch_size, prev_batch, batch_dim=1
            )
            _, stack = utils.manage_step_packed_vars(
                None, stack, batch_size, prev_batch, batch_dim=0
            )

            prev_batch = batch_size
            output, hidden, stack = self(
                input_entry.unsqueeze(0), hidden, stack
            )
            output = self.output_layer(output).squeeze()
            if len(output.shape) < 2:
                output = output.unsqueeze(0)
            loss += self.criterion(output, target_entry)
            outputs.append(torch.argmax(output, -1))
        self.outputs = utils.packed_to_padded(outputs, target_seq_packed)
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
        batch_size = latent_z.shape[1]
        self._update_batch_size(batch_size)

        latent_z = latent_z.repeat(self.n_layers, 1, 1)

        hidden = self.latent_to_hidden(latent_z)
        stack = self.init_stack

        generated_seq = prime_input.repeat(batch_size, 1)
        prime_input = generated_seq.transpose(1, 0).unsqueeze(1)

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

        for idx in range(generate_len):
            if not is_beam:
                output, hidden, stack = self(input_token, hidden, stack)

                logits = self.output_layer(output).squeeze()
                top_idx = search.step(logits)

                input_token = top_idx.view(1, -1).to(self.device)

                generated_seq = torch.cat((generated_seq, top_idx), dim=1)

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
                logits = torch.stack(
                    [self.output_layer(o).squeeze() for o in output]
                )
                hidden = torch.stack(hidden)
                stack = torch.stack(stack)
                input_token, beams = search.step(logits.detach().cpu(), beams)
                input_token = input_token.unsqueeze(1)
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
        return eps.mul(std).add_(mu)  # if self.training else mu

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
            takewhile(lambda x: x != end_token.cpu(), molecule[1:])
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
