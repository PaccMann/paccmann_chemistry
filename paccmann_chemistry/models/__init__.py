from .stack_rnn import StackGRU  # noqa
from .vae import StackGRUEncoder, StackGRUDecoder, TeacherVAE  # noqa
from .encoders import ConvEncoder, GentrlGRUEncoder
from .decoders import nBRCDecoder, DilConvDecoder

ENCODER_FACTORY = {
    'stack': StackGRUEncoder,
    'conv': ConvEncoder,
    'gru': GentrlGRUEncoder
}

DECODER_FACTORY = {
    'stack': StackGRUDecoder,
    'nbrc': nBRCDecoder,
    'conv': DilConvDecoder
}
