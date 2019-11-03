"""Model Parameters Module."""
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from .utils import Identity

OPTIMIZER_FACTORY = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}

ACTIVATION_FACTORY = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'selu': F.selu,
    'tanh': F.tanh,
    'lrelu': nn.LeakyReLU(),
    'elu': F.elu,
    'identity': Identity()
}
