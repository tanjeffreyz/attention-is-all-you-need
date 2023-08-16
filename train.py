import torch
import config
from data import Dataset
from models import Transformer


dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Initialize model with same seed every time
torch.manual_seed(config.SEED)
model = Transformer(config.D_MODEL, len(dataset.src_vocab), len(dataset.trg_vocab))
torch.seed()        # Reseed afterward b/c want shuffled data

print(next(iter(dataset.train_loader)))
test = next(iter(dataset.train_loader))
src = test['source']
trg = test['target']
print(src.size())
print(' '.join(dataset.src_vocab.lookup_tokens(list(src[0]))))
print(' '.join(dataset.trg_vocab.lookup_tokens(list(trg[0]))))
