"""Utilities functions."""
import copy
import logging
import math

import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli

import pytoda
from pytoda.transforms import Compose

logger = logging.getLogger(__name__)


def get_device():
    return torch.device('cuda' if cuda() else 'cpu')


def cuda():
    return torch.cuda.is_available()


DEVICE = get_device()


def sequential_data_preparation(
    input_batch,
    input_keep=1,
    start_index=2,
    end_index=3,
    dropout_index=1,
    device=get_device()
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.
        device (torch.device): Device to be used.
    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor): encoder_seq, decoder_seq,
        target_seq
        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size]`.
    """
    batch_size = input_batch.shape[1]
    input_batch = input_batch.long().to(device)
    decoder_batch = input_batch.clone()
    # apply token dropout if keep != 1
    if input_keep != 1:
        # build dropout indices consisting of dropout_index
        dropout_indices = torch.LongTensor(
            dropout_index * torch.ones(1, batch_size).numpy()
        )
        # mask for token dropout
        mask = Bernoulli(input_keep).sample((input_batch.shape[0], ))
        mask = torch.LongTensor(mask.numpy())
        dropout_loc = np.where(mask == 0)[0]

        decoder_batch[dropout_loc] = dropout_indices

    end_padding = torch.LongTensor(torch.zeros(1, batch_size).numpy())
    target_seq = torch.cat((input_batch[1:, :], end_padding), dim=0)
    target_seq = copy.deepcopy(target_seq).to(device)

    return input_batch, decoder_batch, target_seq


def packed_sequential_data_preparation(
    input_batch,
    input_keep=1,
    start_index=2,
    end_index=3,
    dropout_index=1,
    device=get_device()
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
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

    def _process_sample(sample):
        if len(sample.shape) != 1:
            raise ValueError
        input = sample.long().to(device)
        decoder = input.clone()

        # apply token dropout if keep != 1
        if input_keep != 1:
            # mask for token dropout
            mask = Bernoulli(input_keep).sample(input.shape)
            mask = torch.LongTensor(mask.numpy())
            dropout_loc = np.where(mask == 0)[0]
            decoder[dropout_loc] = dropout_index

        # just .clone() propagates to graph
        target = torch.cat(
            [input[1:].detach().clone(),
             torch.Tensor([0]).long()]
        )
        return input, decoder, target.to(device)

    batch = [_process_sample(sample) for sample in input_batch]

    encoder_decoder_target = zip(*batch)
    encoder_decoder_target = [
        torch.nn.utils.rnn.pack_sequence(entry)
        for entry in encoder_decoder_target
    ]
    return encoder_decoder_target


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


def packed_to_padded(seq, target_packed):
    """Converts a sequence of packed outputs into a padded tensor

    Arguments:
        seq {list} -- List of lists of length T (longest sequence) where each
            sub-list contains output tokens for relevant samples.
        E.g. [len(s) for s in seq] == [8, 8, 8, 4, 3, 1] if batch has 8 samples
        longest sequence has length 6 and only 3/8 samples have length 3.
        target_packed {list} -- Packed target sequence
    Return:
        torch.Tensor: Shape bs x T (padded with 0)

    NOTE:
        Assumes that padding index is 0 and stop_index is 3
    """
    T = len(seq)
    batch_size = len(seq[0])
    padded = torch.zeros(batch_size, T)

    stopped_idx = []
    target_packed += [torch.Tensor()]
    # Loop over tokens per time step
    for t in range(T):
        seq_lst = seq[t].tolist()
        tg_lst = target_packed[t - 1].tolist()

        # Insert Padding token where necessary
        [seq_lst.insert(idx, 0) for idx in sorted(stopped_idx, reverse=False)]
        padded[:, t] = torch.Tensor(seq_lst).long()

        stop_idx = list(filter(lambda x: tg_lst[x] == 3, range(len(tg_lst))))
        stopped_idx += stop_idx

    return padded


def unpack_sequence(seq):
    tensor_seqs, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(seq)
    return [s[:l] for s, l in zip(tensor_seqs.unbind(dim=1), seq_lens)]


def repack_sequence(seq):
    return torch.nn.utils.rnn.pack_sequence(seq)


def perpare_packed_input(input):
    batch_sizes = input.batch_sizes
    data = []
    prev_size = 0
    for batch in batch_sizes:
        size = prev_size + batch
        data.append(input.data[prev_size:size])
        prev_size = size
    return data, batch_sizes


def manage_step_packed_vars(final_var, var, batch_size, prev_batch, batch_dim):
    if batch_size < prev_batch:
        finished_lines = prev_batch - batch_size
        break_index = var.shape[batch_dim] - finished_lines.item()
        finished_slice = slice(break_index, var.shape[batch_dim])
        # var shape: num_layers, batch, cell_size ?
        if batch_dim == 0:
            if final_var is not None:
                final_var[finished_slice, :, :] = var[finished_slice, :, :]
            var = var[:break_index, :, :]
        elif batch_dim == 1:
            if final_var is not None:
                final_var[:, finished_slice, :] = var[:, finished_slice, :]
            var = var[:, :break_index, :]
        else:
            raise ValueError('Allowed batch dim are 1 and 2')

    return final_var, var


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


def to_np(x):
    return x.data.cpu().numpy()


def crop_start_stop(smiles, smiles_language):
    """
    Arguments:
        smiles {torch.Tensor} -- Shape 1 x T
    Returns:
        smiles {torch.Tensor} -- Cropped away everything outside Start/Stop.
    """
    smiles = smiles.tolist()
    try:
        start_idx = smiles.index(smiles_language.start_index)
        stop_idx = smiles.index(smiles_language.stop_index)
        return smiles[start_idx + 1:stop_idx]
    except Exception:
        return smiles


def print_example_reconstruction(reconstruction, inp, language, selfies=False):
    """[summary]

    Arguments:
        reconstruction {[type]} -- [description]
        inp {[type]} -- [description]
        language -- SMILES or ProteinLanguage object
    Raises:
        TypeError: [description]

    Returns:
        [type] -- [description]
    """
    if isinstance(language, pytoda.smiles.SMILESLanguage):
        _fn = language.token_indexes_to_smiles
        if selfies:
            _fn = Compose([_fn, language.selfies_to_smiles])
    elif isinstance(language, pytoda.proteins.ProteinLanguage):
        _fn = language.token_indexes_to_sequence
    else:
        raise TypeError(f'Unknown language class: {type(language)}')

    sample_idx = np.random.randint(len(reconstruction))

    reconstructed = crop_start_stop(reconstruction[sample_idx], language)

    # In padding mode input is tensor
    if isinstance(inp, torch.Tensor):
        inp = inp.permute(1, 0)
        sample = inp[sample_idx].tolist()
    elif isinstance(inp, list):
        sample = inp[sample_idx]

    pred = _fn(reconstructed)
    target = _fn(sample)

    return f'Sample input:\n\t {target}, model reconstructed:\n\t {pred}'
