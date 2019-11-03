"""Loss Functions."""
import torch
from .utils import kl_weight


def vae_loss_function(
    decoder_loss, mu, logvar, kl_growth=0.0015, step=None, eval_mode=False
):
    """
    Loss Function from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114

    Args:
    decoder_loss (float): reconstruction cross-entropy loss over the entire
            sequence of the input
    mu:                 the latent mean, mu
    logvar:             log of the latent variance
    kl_growth (float):  the rate at which the weight grows. Defaults to
        0.0015 resulting in a weight of 1 around step=9000
    step (int):          global train step
    eval_mode (bool): set to True for model evaluation during test and
        validation. Defaults to False for model training.

    Returns: The VAE loss consisting of the cross-entropy (decoder)
    loss and KL divergence (encoder) loss
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if eval_mode:
        kl_w = 1
    else:
        kl_w = kl_weight(step, growth_rate=kl_growth)

    return kl_w * kl_div + decoder_loss, kl_div
