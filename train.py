import torch
import config
from data import train_loader, test_loader, src_vocab, trg_vocab
from models import Embedding


# Initialize model with same seed every time
torch.manual_seed(config.SEED)
src_embedding = Embedding(src_vocab)
torch.seed()        # Reseed afterward b/c want shuffled data

print(next(iter(train_loader)))
src = next(iter(train_loader))['source']
print(src.size())
print(src_embedding(src))
print(src_vocab.lookup_tokens(list(src[0])))
