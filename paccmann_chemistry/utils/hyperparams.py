"""Model Parameters Module."""
import torch.optim as optim
from .search import SamplingSearch, GreedySearch, BeamSearch

SEARCH_FACTORY = {
    'sampling': SamplingSearch,
    'greedy': GreedySearch,
    'beam': BeamSearch,
}

OPTIMIZER_FACTORY = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}
