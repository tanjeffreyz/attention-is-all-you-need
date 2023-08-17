import torch
import config
from data import Dataset
from models import Transformer


print(f'[~] Using device: {Transformer.device}')

dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)
