import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LANGUAGE_PAIR = ('en', 'de')

SOS = '<sos>'
EOS = '<eos>'

BATCH_SIZE = 64
