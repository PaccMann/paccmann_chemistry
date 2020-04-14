"""Train and Test Functions and Utilities."""
import json
import os
from time import time

import torch
from ..utils.search import SamplingSearch
from ..utils.hyperparams import OPTIMIZER_FACTORY
from ..utils.loss_functions import vae_loss_function
from ..utils import get_device, sequential_data_preparation, packed_sequential_data_preparation


def test_vae(model, dataloader, logger, input_keep):
    """
    VAE test function.

    Args:
        model: Model object to be tested.
        dataloader (DataLoader): DataLoader object returning test data batches.
        logger (logging.Logger): To display information on the fly.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.

    Returns:
        float: average test loss over the entire test data.
    """
    device = get_device()
    vae_model = model.to(device)
    vae_model.eval()
    test_loss, test_rec, test_kl_div = 0, 0, 0
    with torch.no_grad():
        for _iter, batch in enumerate(dataloader):
            if (_iter + 1) % 500 == 0:
                logger.info(
                    f'**TESTING**\t Processing batch {_iter}/{len(dataloader)}'
                )
            padded_batch = torch.nn.utils.rnn.pad_sequence(batch)
            padded_batch = padded_batch.to(device)
            encoder_seq, decoder_seq, target_seq = sequential_data_preparation(
                padded_batch,
                input_keep=input_keep,
                start_index=dataloader.dataset.smiles_language.start_index,
                end_index=dataloader.dataset.smiles_language.stop_index,
                dropout_index=dataloader.dataset.smiles_language.unknown_index
            )

            decoder_loss, mu, logvar = vae_model(
                encoder_seq, decoder_seq, target_seq
            )
            loss, kl_div = vae_loss_function(
                decoder_loss, mu, logvar, eval_mode=True
            )
            test_loss += loss.item()
            test_rec += decoder_loss.item()
            test_kl_div += kl_div.item()
    test_loss /= len(dataloader)
    test_rec /= len(dataloader)
    test_kl_div /= len(dataloader)
    vae_model.train()
    return test_loss, test_rec, test_kl_div


def train_vae(
    epoch, model, train_dataloader, val_dataloader, smiles_language,
    model_dir, search=SamplingSearch(), optimizer='adam', lr=1e-3,
    kl_growth=0.0015, input_keep=1., test_input_keep=0., generate_len=100,
    log_interval=100, eval_interval=200, save_interval=200, loss_tracker=None,
    logger=None
):  # yapf: disable
    """
    VAE train function.

    Args:
        epoch (int): Epoch number.
        model: Model object to train.
        train_dataloader (DataLoader): DataLoader object returning
            training batches.
        val_dataloader (DataLoader): DataLoader object returning
            validation batches.
        smiles_language (SMILESLanguage): SMILESLanguage object.
        model_dir (str): The path to the directory where model will
            be saved.
        search (paccmann_chemistry.utils.search.Search): search strategy
                used in the decoder.
        optimizer (str): Choice from OPTIMIZER_FACTORY. Defaults to 'adam'.
        lr (float): The learning rate.
        kl_growth (float): The rate at which the weight grows.
            Defaults to 0.0015 resulting in a weight of 1 around step=9000.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        test_input_keep (float): Like the input_keep parameter, but for
            test. Defaults to 0.
        generate_len (int): Length of the generated molecule.
        log_interval (int): The interval at which average loss is
            recorded.
        eval_interval (int): The interval at which a molecule is generated
            and displayed.
        save_interval (int): The interval at which the model is saved.
        loss_tracker (dict): At each log_interval, update improved test
            losses and respective epoch.
        logger (logging.Logger): To display information on the fly.

    Returns:
         dict: updated loss_tracker.
    """
    if loss_tracker is None:
        loss_tracker = {
            'test_loss_a': 10e4,
            'test_rec_a': 10e4,
            'test_kld_a': 10e4,
            'ep_loss': 0,
            'ep_rec': 0,
            'ep_kld': 0
        }

    device = get_device()
    vae_model = model.to(device)
    vae_model.train()
    train_loss = 0
    optimizer = OPTIMIZER_FACTORY[optimizer](vae_model.parameters(), lr=lr)
    t = time()
    for _iter, batch in enumerate(train_dataloader):

        global_step = (epoch - 1) * len(train_dataloader) + _iter
        # padded_batch = torch.nn.utils.rnn.pad_sequence(batch)
        # packed_batck = torch.nn.utils.rnn.pack_sequence(batch)
        # padded_batch = padded_batch.to(device)

        # WIP: We will use packed sequences instead of padded sequences
        (encoder_seq, decoder_seq,
         target_seq) = packed_sequential_data_preparation(
             batch,
             input_keep=input_keep,
             start_index=start_index,
             end_index=end_index
         )
        optimizer.zero_grad()
        decoder_loss, mu, logvar = vae_model(
            encoder_seq, decoder_seq, target_seq
        )
        loss, kl_div = vae_loss_function(
            decoder_loss, mu, logvar, kl_growth=kl_growth, step=global_step
        )
        loss.backward()
        train_loss += loss.detach().item()
        # for name, param in vae_model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print(param.grad)

        optimizer.step()
        torch.cuda.empty_cache()
        if _iter and _iter % log_interval == 0:
            logger.info(
                f'***TRAINING***\t Epoch: {epoch}, '
                f'step {_iter}/{len(train_dataloader)}.\t'
                f'Loss: {train_loss/log_interval:2.4f}, time spent: {time()-t}'
            )
            t = time()
            train_loss = 0
        if _iter and _iter % save_interval == 0:
            save_dir = os.path.join(
                model_dir, f'weights/saved_model_epoch_{epoch}_iter_{_iter}.pt'
            )
            vae_model.save(save_dir)
            logger.info(f'***SAVING***\t Epoch {epoch}, saved model.')
        if _iter and _iter % eval_interval == 0:
            latent_z = torch.randn(1, mu.shape[0], mu.shape[1]).to(device)
            molecule_iter = vae_model.generate(
                latent_z,
                prime_input=torch.tensor(
                    [train_dataloader.dataset.smiles_language.start_index]
                ).to(device),
                end_token=torch.tensor(
                    [train_dataloader.dataset.smiles_language.stop_index]
                ).to(device),
                generate_len=generate_len,
                search=search
            )
            mol = next(molecule_iter).tolist()
            mol = smiles_language.token_indexes_to_smiles(mol)
            # SELFIES conversion if necessary
            mol = (
                smiles_language.selfies_to_smiles(mol)
                if train_dataloader.dataset._dataset.selfies else mol
            )
            logger.info(f'\nSample Generated Molecule:\n{mol}')

            test_loss, test_rec, test_kld = test_vae(
                vae_model, val_dataloader, logger, test_input_keep
            )
            logger.info(
                f'***TESTING*** \t Epoch {epoch}, test loss = '
                f'{test_loss:.4f}, reconstruction = {test_rec:.4f}, '
                f'KL = {test_kld:.4f}.'
            )

            if test_loss < loss_tracker['test_loss_a']:
                loss_tracker.update(
                    {
                        'test_loss_a': test_loss,
                        'ep_loss': epoch
                    }
                )
                vae_model.save(
                    os.path.join(model_dir, f'weights/best_loss.pt')
                )
                logger.info(
                    f'Epoch {epoch}. NEW best test loss = {test_loss:.4f} \t'
                    f'(Rec = {test_rec:.4f}, KLD = {test_kld:.4f}).'
                )

            if test_rec < loss_tracker['test_rec_a']:
                loss_tracker.update({'test_rec_a': test_rec, 'ep_rec': epoch})
                vae_model.save(os.path.join(model_dir, f'weights/best_rec.pt'))
                logger.info(
                    f'Epoch {epoch}. NEW best reconstruction loss = '
                    f'{test_rec:.4f} \t (Loss = {test_loss:.4f}, KLD = '
                    f'{test_kld:.4f})'
                )
            if test_kld < loss_tracker['test_kld_a']:
                loss_tracker.update({'test_kld_a': test_kld, 'ep_kld': epoch})
                vae_model.save(os.path.join(model_dir, f'weights/best_kld.pt'))
                logger.info(
                    f'Epoch {epoch}. NEW best KLD = {test_kld:.4f} \t (loss '
                    f'= {test_loss:.4f}, Reconstruction = {test_rec:.4f}).'
                )
            with open(os.path.join(model_dir, 'loss_tracker.json'), 'w') as fp:
                json.dump(loss_tracker, fp)

    logger.info(
        f'Epoch {epoch} finished, \t Training Loss = {loss.item():.4f},'
        f'Reconstruction = {decoder_loss.item():.4f}, KL = '
        f'{(loss - decoder_loss).item():.8f}'
    )

    return loss_tracker
