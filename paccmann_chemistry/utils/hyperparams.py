"""Model Parameters Module."""
import torch.optim as optim
from .search import SamplingSearch, GreedySearch, BeamSearch

SEARCH_FACTORY = {
    'Sampling': SamplingSearch,
    'Greedy': GreedySearch,
    'Beam': BeamSearch,
}

OPTIMIZER_FACTORY = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}
