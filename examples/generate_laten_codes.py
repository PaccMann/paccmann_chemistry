"""Generate latent codes using the paccmann model."""
import json
import logging
import os
import sys
from time import time
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from paccmann_chemistry.utils import collate_fn, get_device
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.models.training import (
    train_vae, get_data_preparation
)
from paccmann_chemistry.utils.hyperparams import SEARCH_FACTORY
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
import torch

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('training_vae')

# %%
BASE_PATH = os.path.expanduser(
    '~/Box/Molecular_SysBio/data/'
    'paccmann/paccmann_affinity/'
    'trained_models/vae_selfies_one_hot'
)
PARAMS_FILE = os.path.join(BASE_PATH, 'model_params.json')  # Add model dir
MODEL_PATH = os.path.join(
    BASE_PATH, 'weights', 'best_loss.pt'
)  # Add model dir
SAMPLES_PATH = os.path.expanduser(
    '~/Box/Molecular_SysBio/data/'
    'paccmann/paccmann_affinity/'
    'all_molecules.smi'
)
SMILES_LANGUAGE_PATH = os.path.join(BASE_PATH, 'selfies_language.pkl')


# %%
def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()

device = get_device()
# read the params json
params = dict()
with open(PARAMS_FILE, 'r') as f:
    params.update(json.load(f))

params['batch_size'] = 1

# get params
model_dir = MODEL_PATH

# Load SMILES language
smiles_language = SMILESLanguage.load(SMILES_LANGUAGE_PATH)

data_preparation = get_data_preparation(params.get('batch_mode'))
device = get_device()

dataset = SMILESDataset(
    SAMPLES_PATH,
    smiles_language=smiles_language,
    padding=False,
    selfies=params.get('selfies', False),
    add_start_and_stop=params.get('add_start_stop_token', True),
    augment=False,  #params.get('augment_smiles', False),
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
    drop_last=True,
    shuffle=True,
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

gru_vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))


# Updating the vocab size will break the model
params.update({
    # 'vocab_size': smiles_language.number_of_tokens,
    'pad_index': smiles_language.padding_index
})  # yapf:disable

# if params.get('embedding', 'learned') == 'one_hot':
#     params.update({'embedding_size': params['vocab_size']})

# train for n_epoch epochs
logger.info(
    'Model creation, laoding and data processing done. '
    'Evaluation starts.'
)

decoder_search = SEARCH_FACTORY[
    params.get('decoder_search', 'sampling')
](
    temperature=params.get('temperature', 1.),
    beam_width=params.get('beam_width', 3),
    top_tokens=params.get('top_tokens', 5)
)  # yapf: disable


# %%

# %%
gru_vae.eval()
gru_vae.to(device)

# %%
counter = 0
with torch.no_grad():
    latent_code = []
    from tqdm import tqdm
    for batch in tqdm(dataloader, total=len(dataloader)):
        (encoder_seq, _, _) = data_preparation(
            batch, input_keep=0., start_index=2, end_index=3, device=device
        )
        try:
            mu, logvar = gru_vae.encode(encoder_seq)
        except RuntimeError:
            # Catch the error and substitute any new tokens by "<UNK>" tokens
            new_seq = []
            padd_encoder_seq, lenghts = torch.nn.utils.rnn.pad_packed_sequence(
                encoder_seq, batch_first=True
            )
            for seq, _len in zip(padd_encoder_seq, lenghts):
                seq = seq[:_len]
                if any([x >= params['vocab_size'] for x in seq]):
                    seq = torch.tensor(
                        [
                            x if x < params['vocab_size'] else
                            smiles_language.unknown_index for x in seq.tolist()
                        ]
                    ).short()

                    failed_smiles = smiles_language.selfies_to_smiles(
                        smiles_language.token_indexes_to_smiles(seq.tolist())
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
        for _mu in mu.tolist():
            latent_code.append([counter, _mu])
            counter += 1

LATENT_CODE_PATH = os.path.join(
    os.path.dirname(SAMPLES_PATH), 'samples_latent_code.tsv'
)

with open(LATENT_CODE_PATH, 'w') as f:
    for i, mu in latent_code:
        f.write(f'{i}\t{",".join([str(x) for x in mu[0]])}\n')
