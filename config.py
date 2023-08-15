import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Language settings
LANGUAGE_PAIR = ('en', 'de')
SOS = '<sos>'
EOS = '<eos>'

# Data dimensions
BATCH_SIZE = 64
D_MODEL = 512
