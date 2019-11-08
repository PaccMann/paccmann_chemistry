"""Stack Augmented GRU Implementation."""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import get_device
from ..utils.hyperparams import OPTIMIZER_FACTORY


class StackGRU(nn.Module):
    """Stack Augmented Gated Recurrent Unit (GRU) class."""

    def __init__(self, params):
        """
        Initialization.

        Reference:
            GRU layers intended to help with the training of VAEs by weakening
            the decoder as proposed in: https://arxiv.org/abs/1511.06349.

        Args:
            params (dict): Hyperparameters.

        Items in params:
            input_size (int): Vocabulary size.
            embedding_size (int): The embedding size for the dict tokens
            rnn_cell_size (int): Hidden size of GRU.
            output_size (int): Output size of GRU (vocab size).
            stack_width (int): Number of stacks in parallel.
            stack_depth (int): Stack depth.
            n_layers (int): The number of GRU layer.
            dropout (float): Dropout on the output of GRU layers except the
                last layer.
            batch_size (int): Batch size.
            lr (float, optional): Learning rate default 0.01.
            optimizer (str, optional): Choice from OPTIMIZER_FACTORY.
                Defaults to 'Adadelta'.
            padding_index (int, optional): Index of the padding token.
                Defaults to 0.
            bidirectional (bool, optional): Whether to train a bidirectional
                GRU. Defaults to False.
        """

        super(StackGRU, self).__init__()

        self.input_size = params['input_size']
        self.embedding_size = params['embedding_size']
        self.rnn_cell_size = params['rnn_cell_size']
        self.output_size = params['output_size']
        self.stack_width = params['stack_width']
        self.stack_depth = params['stack_depth']
        self.batch_size = params['batch_size']
        self.use_cuda = torch.cuda.is_available()
        self.n_layers = params['n_layers']
        self.bidirectional = params.get('bidirectional', False)
        self.n_directions = 2 if self.bidirectional else 1
        self.device = get_device()

        # Network
        self.stack_controls_layer = nn.Linear(
            in_features=self.rnn_cell_size, out_features=3
        )
        self.stack_input_layer = nn.Linear(
            in_features=self.rnn_cell_size, out_features=self.stack_width
        )

        self.encoder = nn.Embedding(
            self.input_size,
            self.embedding_size,
            padding_idx=params.get('pad_index', 0)
        )
        self.gru = nn.GRU(
            self.embedding_size + self.stack_width,
            self.rnn_cell_size,
            self.n_layers,
            bidirectional=self.bidirectional,
            dropout=params['dropout']
        )
        self.decoder = nn.Linear(
            self.rnn_cell_size * self.n_directions, self.output_size
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OPTIMIZER_FACTORY[
            params.get('optimizer', 'Adadelta')
        ](self.parameters(), lr=params.get('lr', 0.01))  # yapf: disable

        self._check_params()

    def forward(self, input_token, hidden, stack):
        """
        StackGRU forward function.

        Args:
            input_token (torch.Tensor): LongTensor containing
                indices of the input token of size `[1, batch_size]`.
            hidden (torch.Tensor): Hidden state of size
                `[n_layers*n_directions, batch_size, rnn_cell_size]`.
            stack (torch.Tensor): Previous step's stack of size
                `[batch_size, stack_depth, stack_width]`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): output, hidden, stack.

            Output of size `[batch_size, output_size]`.
            Hidden state of size `[1, batch_size, rnn_cell_size]`.
            Stack of size `[batch_size, stack_depth, stack_width]`.
        """
        if input_token.shape[0] != 1:
            input_token = input_token.view(1, -1)
        embedded_input = self.encoder(input_token.to(self.device))
        stack_controls = self.stack_controls_layer(
            hidden[-1, :, :].unsqueeze(0)
        )
        stack_controls = F.softmax(stack_controls, dim=1)
        stack_input = self.stack_input_layer(hidden[-1, :, :].unsqueeze(0))
        stack_input = torch.tanh(stack_input)
        stack = self.stack_augmentation(
            stack_input.permute(1, 0, 2), stack, stack_controls
        )
        stack_top = stack[:, 0, :].unsqueeze(0)
        inp = torch.cat((embedded_input, stack_top), dim=2)
        output, hidden = self.gru(inp, hidden)
        output = self.decoder(output).squeeze()
        return output, hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        """
        Stack update function.

        Args:
            input_val (torch.Tensor): Contributon of the current
                hidden state to be input to the stack.
                Must be of shape `[batch_size, 1, stack_width]`.
            prev_stack (torch.Tensor): The stack from previous
                step. Must be of shape
                `[batch_size, stack_depth, stack_width]`.
            controls (torch.Tensor): Stack controls giving
                probabilities of PUSH, POP or NO-OP for the pushdown
                stack. Must be of shape `[batch_size, 3]`.

        Returns:
            torch.Tensor: Updated stack of shape
                `[batch_size, stack_depth, stack_width]`.
        """
        batch_size = prev_stack.size(0)
        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda())
        else:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = (
            controls[:, 0], controls[:, 1], controls[:, 2]
        )
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack
        new_stack = new_stack + a_push * stack_up
        new_stack = new_stack + a_pop * stack_down
        return new_stack

    def init_hidden(self, batch_size=None):
        """Initializes hidden state."""
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_cuda:
            return Variable(
                torch.zeros(
                    self.n_layers * self.n_directions, batch_size,
                    self.rnn_cell_size
                ).cuda()
            )

        return Variable(
            torch.zeros(
                self.n_layers * self.n_directions, batch_size,
                self.rnn_cell_size
            )
        )

    def init_stack(self, batch_size=None):
        """Initializes Stack."""
        if batch_size is None:
            batch_size = self.batch_size
        result = torch.zeros(batch_size, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())

        return Variable(result)

    def train_step(self, input_seq, target_seq):
        """
        The train step method.

        Args:
            input_seq (torch.Tensor): Padded tensor of indices for a batch of
                input sequences. Must be of shape
                `[max batch sequence length +1, batch_size]`.
            target_seq (torch.Tensor): Padded tensor of indices for a batch of
                target sequences. Must be of shape
                `[max batch sequence length +1, batch_size]`.

        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.

        Returns:
            float: Average loss for all sequence steps.
        """
        hidden = self.init_hidden()
        stack = self.init_stack()
        self.zero_grad()
        loss = 0
        for input_entry, target_entry in zip(input_seq, target_seq):
            output, hidden, stack = self(input_entry, hidden, stack)
            loss += self.criterion(
                output, torch.LongTensor(target_entry.squeeze().numpy())
            )
        loss.backward()
        self.optimizer.step()

        return loss.item() / len(input_seq)

    def generate(
        self, prime_input, end_token, generate_len=100, temperature=0.8
    ):
        """
        The evaluation method.

        Args:
            prime_input (torch.tensor): Indices for the priming string of size
                `[batch_size, length of priming sequences]`.
                Example: `prime_input = torch.tensor([[2, 4, 5], [2, 4, 7]])`
            end_token (torch.tensor): End token for the generated molecule.
            generate_len (int): Length of the generated molecule.
            temperature (float): Softmax temperature parameter between 0 and 1.
                Lower temperatures result in a more descriminative softmax.

        Returns:
            The sequence of indices for the generated molecule.
        """
        prime_input = prime_input.transpose(1, 0).view(-1, 1, len(prime_input))
        batch_size = prime_input.shape[-1]
        hidden = self.init_hidden(batch_size)
        stack = self.init_stack(batch_size)
        generated_seq = prime_input.transpose(0, 2)
        # Use priming string to "build up" hidden state
        for prime_entry in prime_input[:-1]:
            _, hidden, stack = self(prime_entry, hidden, stack)
        input_token = prime_input[-1]

        for _ in range(generate_len):
            output, hidden, stack = self(input_token, hidden, stack)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.cpu().view(batch_size,
                                                 -1).div(temperature).exp()
            top_idx = torch.tensor(
                torch.multinomial(output_dist, 1).cpu().numpy()
            )
            # Add generated_seq character to string and use as next input
            generated_seq = torch.cat(
                (generated_seq, top_idx.unsqueeze(2)), dim=2
            )
            input_token = top_idx.view(1, -1)
            # break when end token is generated
            if batch_size == 1 and top_idx == end_token:
                break
        return generated_seq

    def _check_params(self):
        """
        Runs size checks on input parameter

        """

        if self.rnn_cell_size < self.embedding_size:
            warnings.warn('Refrain from squashing embeddings in RNN cells')
