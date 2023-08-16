import config
from data import Dataset
from models import Transformer


dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Initialize model with same seed every time
model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab)
)

# print(next(iter(dataset.train_loader)))
test = next(iter(dataset.train_loader))
src = test['source']
trg = test['target']
index = min(src.size(-1), trg.size(-1)) - 1
print('IN: ', src.size(), trg.size())
print(' '.join(dataset.src_vocab.lookup_tokens(list(src[index]))))
print(' '.join(dataset.trg_vocab.lookup_tokens(list(trg[index]))))

print('OUT:', model(src, trg).size(), len(dataset.trg_vocab))
