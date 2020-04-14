#!/usr/bin/env python3
"""Train TeacherVAE molecule generator."""
import argparse
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
from paccmann_chemistry.models.training import train_vae
from paccmann_chemistry.utils.hyperparams import SEARCH_FACTORY
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
import torch

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('training_vae')


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


# yapf: disable
parser = argparse.ArgumentParser(description='Chemistry VAE training script.')
parser.add_argument(
    'train_smiles_filepath', type=str,
    help='Path to the train data file (.smi).'
)
parser.add_argument(
    'test_smiles_filepath', type=str,
    help='Path to the test data file (.smi).'
)
parser.add_argument(
    'smiles_language_filepath', type=str,
    help='Path to SMILES language object.'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable


def main(parser_namespace):
    try:
        disable_rdkit_logging()
        # read the params json
        params = dict()
        with open(parser_namespace.params_filepath) as f:
            params.update(json.load(f))

        # get params
        train_smiles_filepath = parser_namespace.train_smiles_filepath
        test_smiles_filepath = parser_namespace.test_smiles_filepath
        smiles_language_filepath = parser_namespace.smiles_language_filepath
        model_path = parser_namespace.model_path
        training_name = parser_namespace.training_name

        logger.info(f'Model with name {training_name} starts.')

        model_dir = os.path.join(model_path, training_name)
        log_path = os.path.join(model_dir, 'logs')
        val_dir = os.path.join(log_path, 'val_logs')
        os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Load SMILES language
        smiles_language = SMILESLanguage.load(smiles_language_filepath)

        params.update({
            'vocab_size': smiles_language.number_of_tokens,
            'pad_index': smiles_language.padding_index
        })  # yapf:disable
        # create SMILES eager dataset
        smiles_train_data = SMILESDataset(
            train_smiles_filepath,
            smiles_language=smiles_language,
            padding=False,
            selfies=params.get('selfies', False),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=params.get('augment_smiles', False),
            canonical=params.get('canonical', False),
            kekulize=params.get('kekulize', False),
            all_bonds_explicit=params.get('all_bonds_explicit', False),
            all_hs_explicit=params.get('all_hs_explicit', False),
            remove_bonddir=params.get('remove_bonddir', False),
            remove_chirality=params.get('remove_chirality', False),
            backend='eager'
        )
        smiles_test_data = SMILESDataset(
            test_smiles_filepath,
            smiles_language=smiles_language,
            padding=False,
            selfies=params.get('selfies', False),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=params.get('augment_smiles', False),
            canonical=params.get('canonical', False),
            kekulize=params.get('kekulize', False),
            all_bonds_explicit=params.get('all_bonds_explicit', False),
            all_hs_explicit=params.get('all_hs_explicit', False),
            remove_bonddir=params.get('remove_bonddir', False),
            remove_chirality=params.get('remove_chirality', False),
            backend='eager'
        )

        # Update the smiles_vocabulary size
        if not params.get('embedding', 'learned') == 'pretrained':
            params.update({'vocab_size': smiles_language.number_of_tokens})

        with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
            json.dump(params, fp)

        # create DataLoaders
        train_data_loader = torch.utils.data.DataLoader(
            smiles_train_data,
            batch_size=params.get('batch_size', 64),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            pin_memory=params.get('pin_memory', True),
            num_workers=params.get('num_workers', 8)
        )

        test_data_loader = torch.utils.data.DataLoader(
            smiles_test_data,
            batch_size=params.get('batch_size', 64),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            pin_memory=params.get('pin_memory', True),
            num_workers=params.get('num_workers', 8)
        )
        # initialize encoder and decoder
        device = get_device()
        gru_encoder = StackGRUEncoder(params).to(device)
        gru_decoder = StackGRUDecoder(params).to(device)
        gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
        loss_tracker = {
            'test_loss_a': 10e4,
            'test_rec_a': 10e4,
            'test_kld_a': 10e4,
            'ep_loss': 0,
            'ep_rec': 0,
            'ep_kld': 0
        }

        # train for n_epoch epochs
        logger.info(
            'Model creation and data processing done, Training starts.'
        )
        decoder_search = SEARCH_FACTORY[
            params.get('decoder_search', 'sampling')
        ](
            temperature=params.get('temperature', 1.),
            beam_width=params.get('beam_width', 3),
            top_tokens=params.get('top_tokens', 5)
        )  # yapf: disable

        for epoch in range(params['epochs'] + 1):
            t = time()
            loss_tracker = train_vae(
                epoch,
                gru_vae,
                train_data_loader,
                test_data_loader,
                smiles_language,
                model_dir,
                search=decoder_search,
                optimizer=params.get('optimizer', 'adadelta'),
                lr=params['learning_rate'],
                kl_growth=params['kl_growth'],
                input_keep=params['input_keep'],
                test_input_keep=params['test_input_keep'],
                generate_len=params['generate_len'],
                log_interval=params['log_interval'],
                save_interval=params['save_interval'],
                eval_interval=params['eval_interval'],
                loss_tracker=loss_tracker,
                logger=logger
            )
            logger.info(f'Epoch {epoch}, took {time() - t:.1f}.')

        logger.info(
            'OVERALL: \t Best loss = {0:.4f} in Ep {1}, '
            'best Rec = {2:.4f} in Ep {3}, '
            'best KLD = {4:.4f} in Ep {5}'.format(
                loss_tracker['test_loss_a'], loss_tracker['ep_loss'],
                loss_tracker['test_rec_a'], loss_tracker['ep_rec'],
                loss_tracker['test_kld_a'], loss_tracker['ep_kld']
            )
        )
        logger.info('Training done, shutting down.')
    except Exception:
        logger.exception('Exception occurred while running train_vae.py.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(parser_namespace=args)
