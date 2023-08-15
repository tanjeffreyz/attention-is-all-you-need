import config
from data import train_loader, test_loader, src_vocab, trg_vocab

import torch
from torch import nn

embedding = nn.Embedding(len(src_vocab), config.D_MODEL)
print(embedding(torch.tensor(next(iter(train_loader))['en'][0])))

print(next(iter(train_loader))['en'][0])
print(src_vocab.lookup_tokens(list(next(iter(train_loader))['en'][0])))
