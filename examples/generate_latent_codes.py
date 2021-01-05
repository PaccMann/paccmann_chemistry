"""Generate latent codes using the paccmann model."""
import json
import logging
import os
import sys
import argparse
import torch
from pytoda.files import read_smi
import pandas as pd

from paccmann_chemistry.models.training import get_data_preparation
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.utils import (
    collate_fn, get_device, disable_rdkit_logging
)
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('latent_code_vae')

parser = argparse.ArgumentParser(description='VAE latent code generation.')
parser.add_argument(
    'model_path', type=str, help='Path to the trained model (SELFIES VAE)'
)
parser.add_argument(
    'data_path', type=str, help='Path to the data file (.smi).'
)


def main(parser_namespace):

    disable_rdkit_logging()

    model_path = parser_namespace.model_path
    data_path = parser_namespace.data_path

    weights_path = os.path.join(model_path, 'weights', 'best_loss.pt')

    device = get_device()
    # read the params json
    params = dict()
    with open(os.path.join(model_path, 'model_params.json'), 'r') as f:
        params.update(json.load(f))

    params['batch_size'] = 1

    # Load SMILES language
    smiles_language = SMILESLanguage.load(
        os.path.join(model_path, 'selfies_language.pkl')
    )

    data_preparation = get_data_preparation(params.get('batch_mode'))
    device = get_device()

    print('Selfies', params.get('selfies', False))

    dataset = SMILESDataset(
        data_path,
        smiles_language=smiles_language,
        padding=False,
        selfies=params.get('selfies', False),
        add_start_and_stop=params.get('add_start_stop_token', True),
        augment=False,
        canonical=params.get('canonical', False),
        kekulize=params.get('kekulize', False),
        all_bonds_explicit=params.get('all_bonds_explicit', False),
        all_hs_explicit=params.get('all_hs_explicit', False),
        remove_bonddir=params.get('remove_bonddir', False),
        remove_chirality=params.get('remove_chirality', False),
        backend='lazy',
        device=device
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.get('batch_size', 64),
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        pin_memory=params.get('pin_memory', True),
        num_workers=params.get('num_workers', 8)
    )
    # initialize encoder and decoder
    gru_encoder = StackGRUEncoder(params).to(device)
    gru_decoder = StackGRUDecoder(params).to(device)
    gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
    logger.info('\n****MODEL SUMMARY***\n')
    for name, parameter in gru_vae.named_parameters():
        logger.info(f'Param {name}, shape:\t{parameter.shape}')
    total_params = sum(p.numel() for p in gru_vae.parameters())
    logger.info(f'Total # params: {total_params}')

    gru_vae.load_state_dict(torch.load(weights_path, map_location=device))

    # Updating the vocab size will break the model
    params.update({
        # 'vocab_size': smiles_language.number_of_tokens,
        'pad_index': smiles_language.padding_index
    })  # yapf:disable

    # if params.get('embedding', 'learned') == 'one_hot':
    #     params.update({'embedding_size': params['vocab_size']})

    # train for n_epoch epochs
    logger.info(
        'Model creation, loading and data processing done. Evaluation starts.'
    )

    gru_vae.eval()
    gru_vae.to(device)
    counter = 0
    with torch.no_grad():
        latent_code = []
        from tqdm import tqdm
        for batch in tqdm(dataloader, total=len(dataloader)):
            (encoder_seq, _, _) = data_preparation(
                batch,
                input_keep=0.,
                start_index=2,
                end_index=3,
                device=device
            )
            try:
                mu, logvar = gru_vae.encode(encoder_seq)
            except RuntimeError:
                # Substitute any new tokens by "<UNK>" tokens
                new_seq = []
                padd_encoder_seq, lenghts = (
                    torch.nn.utils.rnn.pad_packed_sequence(
                        encoder_seq, batch_first=True
                    )
                )
                for seq, _len in zip(padd_encoder_seq, lenghts):
                    seq = seq[:_len]
                    if any([x >= params['vocab_size'] for x in seq]):
                        seq = torch.tensor(
                            [
                                x if x < params['vocab_size'] else
                                smiles_language.unknown_index
                                for x in seq.tolist()
                            ]
                        ).short()

                        failed_smiles = smiles_language.selfies_to_smiles(
                            smiles_language.token_indexes_to_smiles(
                                seq.tolist()
                            )
                        )
                        logger.warning(
                            f'Out of bounds sample: ~{counter}\t{failed_smiles}'
                        )
                    new_seq.append(seq)

                if new_seq:
                    for _ in range(params['batch_size'] - len(new_seq)):
                        new_seq.append(torch.ones_like(new_seq[-1]))
                    (encoder_seq, _, _) = data_preparation(
                        new_seq,
                        input_keep=0.,
                        start_index=2,
                        end_index=3,
                        device=device
                    )
                    mu, logvar = gru_vae.encode(encoder_seq)
            for _mu, _logvar in zip(mu, logvar):
                latent = torch.exp(0.5 * _logvar) + _mu
                latent_code.append([counter, _mu, _logvar, latent])
                counter += 1

    idxs, mus, logvars, latents = zip(*latent_code)
    mu, logvar, latent = (
        torch.stack(mus), torch.stack(logvars), torch.stack(latents)
    )
    data = read_smi(data_path)
    df = pd.DataFrame(
        {
            'Compound': data.index.tolist(),
            'SMILES': data['SMILES'].tolist()
        }
    )
    for var, name in zip([mu.T, logvar.T, latent.T], ['mu', 'logvar', 'z']):
        for idx in range(len(var)):
            df[name + '_' + str(idx)] = var[idx].tolist()

    LATENT_CODE_PATH = os.path.join(
        os.path.dirname(data_path), 'samples_latent_code.tsv'
    )
    df.to_csv(LATENT_CODE_PATH)


if __name__ == '__main__':
    args = parser.parse_args()
    main(parser_namespace=args)
