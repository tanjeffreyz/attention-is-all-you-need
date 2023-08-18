import torch
import config
import os
import numpy as np
from data import Dataset
from models import Transformer
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


print('[~] Training')
print(f' ~  Using device: {Transformer.device}')
writer = SummaryWriter()
now = datetime.now()

dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab),
    dataset.src_vocab[dataset.pad_token],
    dataset.trg_vocab[dataset.pad_token]
)

print(f' ~  Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

loss_function = torch.nn.CrossEntropyLoss()

# Set up experiment folder structure
root = os.path.join(
    'experiments',
    '-'.join(config.LANGUAGE_PAIR),
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
weight_dir = os.path.join(root, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)

# Metrics
train_losses = np.empty((2, 0))
valid_losses = np.empty((2, 0))


def save_metrics():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'valid_losses'), valid_losses)


# Train
print()
for epoch in tqdm(range(config.NUM_EPOCHS), desc='Epoch'):
    model.train()
    train_loss = 0
    num_batches = 0     # Using DataPipe, cannot use len() to get number of batches
    for data in dataset.train_loader:
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)
        trg_oh = torch.nn.functional.one_hot(trg, len(dataset.trg_vocab)).float().to(model.device)

        # Given the sequence length N, transformer tries to predict the N+1th token.
        # Thus, transformer must take in trg[:-1] as input and predict trg[1:] as output.
        optimizer.zero_grad()
        predictions = model(src, trg[:, :-1])
        loss = loss_function(predictions, trg_oh[:, 1:])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1
        del src, trg, trg_oh

    train_loss /= num_batches
    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)

    if epoch % 10 == 0:
        # Evaluate model
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            num_batches = 0
            for data in dataset.valid_loader:
                src = data['source'].to(model.device)
                trg = data['target'].to(model.device)
                trg_oh = torch.nn.functional.one_hot(trg, len(dataset.trg_vocab)).float().to(model.device)

                predictions = model(src, trg[:, :-1])
                loss = loss_function(predictions, trg_oh[:, 1:])

                valid_loss += loss.item()
                num_batches += 1
                del src, trg, trg_oh

            valid_loss /= num_batches
            valid_losses = np.append(valid_losses, [[epoch], [valid_loss]], axis=1)
            writer.add_scalar('Loss/validation', valid_loss, epoch)

            save_metrics()

save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
