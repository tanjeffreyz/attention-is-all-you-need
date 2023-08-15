import config
from data import train_loader, test_loader, src_vocab, trg_vocab, src_embedding, trg_embedding

import torch

src = next(iter(train_loader))['en'][0]
print(src)
print(src_embedding(torch.tensor(src)))
print(src_vocab.lookup_tokens(list(src)))
