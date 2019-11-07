"""Model Parameters Module."""
import torch.optim as optim

OPTIMIZER_FACTORY = {
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'Adamax': optim.Adamax,
    'RMSprop': optim.RMSprop,
    'SGD': optim.SGD
}
