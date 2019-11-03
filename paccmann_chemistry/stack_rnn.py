"""Stack Augmented GRU Implementation."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .hyperparams import OPTIMIZER_FACTORY
from .utils import get_device


class StackGRU(nn.Module):
    """Stack Augmented GRU class."""

    def __init__(self, params):
        """
        Initialization.

        Args:
            params (dict): with hyperparameters.
                This should contain:
                - input_size (int): vocabulary size
                - hidden_size (int): hidden size of GRU
                - output_size (int): output size of GRU (vocab size)
                - stack_width (int): number of stacks in parallel
                - stack_depth (int): stack depth
                - n_layers (int): number of GRU layer
                - optimizer (str): choose from: 'Adadelta', 'Adagrad','Adam', 'Adamax',
                    'RMSprop', 'SGD', defaults to 'Adadelta'.
                - lr (float): learning rate default 0.01
                - padding_index (int): index of the padding token
                - bidirectional (bool): bidirectional GRU, defaults to False
                - batch_size (int): batch size, defaults to 32
                - dropout (float): dropout on the output of GRU layers except the last layer.

        GRU layers intended to help with the training of VAEs by weakening the decoder as
            proposed in: https://arxiv.org/abs/1511.06349
        """

        super(StackGRU, self).__init__()

        self.params = params
        self.bidirectional = params['bidirectional']
        self.n_directions = 2 if self.bidirectional else 1
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['input_size']
        self.stack_width = params['stack_width']
        self.stack_depth = params['stack_depth']
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']
        self.lr = params.get('lr', 0.01)
        self.has_cell = params.get('has_cell', False)
        self.has_stack = params.get('has_stack', True)
        self.use_cuda = torch.cuda.is_available()
        self.n_layers = params['n_layers']
        self.padding_index = params.get('pad_index', 0)
        self.optimizer = params.get('optimizer', 'Adadelta')
        self.activation = params.get('activation', 'relu')
        self.device = get_device()

        # Network
        self.stack_controls_layer = nn.Linear(
            in_features=self.hidden_size, out_features=3
        )
        self.stack_input_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.stack_width
        )

        self.encoder = nn.Embedding(
            self.input_size, self.hidden_size, padding_idx=self.padding_index
        )
        self.gru = nn.GRU(
            self.hidden_size + self.stack_width,
            self.hidden_size,
            self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )
        self.decoder = nn.Linear(
            self.hidden_size * self.n_directions, self.output_size
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OPTIMIZER_FACTORY[
            self.optimizer
        ](self.parameters(), lr=self.lr)    # yapf: disable

    def load_model(self, path):
        """Load Model From Path."""
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        """Save Model to Path."""
        torch.save(self.state_dict(), path)

    def forward(self, input_token, hidden, stack):
        """
        StackGRU forward function.

        Args:
            input_token (torch.Tensor): LongTensor containing
                indices of the input token of size [1, batch_size]
            hidden (torch.Tensor): hidden state of size
                [n_layers*n_directions, batch_size, hidden_size]
            stack (torch.Tensor): previous step's stack of size
                [batch_size, stack_depth, stack_width]

        Returns:
            output (torch.Tensor): output of size
                [batch_size, output_size]
            hidden (torch.Tensor): hidden state of size
                [1, batch_size, hidden_size]
            stack (torch.Tensor): stack of size
                [batch_size, stack_depth, stack_width]
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
            input_val (torch.Tensor): contributon of the current
                hidden state to be input to the stack.
                Must be of shape: [batch_size, 1, stack_width].
            prev_stack (torch.Tensor): the stack from previous
                step. Must be of shape:
                [batch_size, stack_depth, stack_width].
            controls (torch.Tensor): stack controls giving
                probabilities of PUSH, POP or NO-OP for the pushdown
                stack. Must be of shape: [batch_size, 3].

        Returns:
            new_stack (torch.Tensor): updated stack of shape:
                [batch_size, stack_depth, stack_width].
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
                    self.hidden_size
                ).cuda()
            )
        else:
            return Variable(
                torch.zeros(
                    self.n_layers * self.n_directions, batch_size,
                    self.hidden_size
                )
            )

    def init_stack(self, batch_size=None):
        """Initializes Stack."""
        if batch_size is None:
            batch_size = self.batch_size
        result = torch.zeros(batch_size, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, input_seq, target_seq):
        """
        The train step method.

        Args:
            input_seq (torch.Tensor): padded tensor of indices for a batch of input 
                sequences. Must be of shape: [max batch sequence length +1, batch_size].
            target_seq (torch.Tensor): padded tensor of indices for a batch of 
                target sequences. Must be of shape: 
                [max batch sequence length +1, batch_size].

            Note: input and target sequences are outputs of seq_data_prep(batch) 
                with batches returned by a DataLoader object.

        Returns:
            Average loss for all sequence steps.
        """
        hidden = self.init_hidden()
        stack = self.init_stack()
        self.zero_grad()
        loss = 0
        for idx in range(len(input_seq)):
            output, hidden, stack = self(input_seq[idx], hidden, stack)
            loss += self.criterion(
                output, torch.LongTensor(target_seq[idx].squeeze().numpy())
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
            prime_input (torch.tensor): tensor of indices
                for the priming string of size
                [batch_size, length of priming sequences].

            Example:
                prime_input = [[2, 4, 5], [2, 4, 7]]
                prime_input = torch.tensor(prime_input)

            end_token (torch.tensor): end token for the
                generated molecule.
            generate_len (int): length of the generated molecule
            temperature (float): softmax temperature parameter
                between 0 and 1. Lower temperatures result in a
                more descriminative softmax.

        Returns:
            The sequence of indices for the generated molecule.
        """
        prime_input = prime_input.transpose(1, 0).view(-1, 1, len(prime_input))
        batch_size = prime_input.shape[-1]
        hidden = self.init_hidden(batch_size)
        stack = self.init_stack(batch_size)
        generated_seq = prime_input.transpose(0, 2)
        # Use priming string to "build up" hidden state
        for p in range(len(prime_input) - 1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        input_token = prime_input[-1]

        for p in range(generate_len):
            output, hidden, stack = self.forward(input_token, hidden, stack)

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
