import torch
import config
import os
import numpy as np
from data import Dataset
from models import Transformer
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


print('[~] Training:')
print(f' ~  Using device: {Transformer.device}')
writer = SummaryWriter()
now = datetime.now()

dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab)
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
test_losses = np.empty((2, 0))


def save_metrics():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'test_losses'), test_losses)


# Train
print()
for epoch in tqdm(range(config.NUM_EPOCHS), desc='Epoch'):
    model.train()
    train_loss = 0
    num_batches = 0     # Using DataPipe, cannot retrieve length beforehand
    for data in dataset.train_loader:
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)

        # Transformer predicts probabilities
        # Ground truth is one-hot encoding of the target vocabulary words
        trg_one_hot = torch.nn.functional.one_hot(trg[:, 1:], len(dataset.trg_vocab)).float()

        optimizer.zero_grad()
        predictions = model(src, trg[:, :-1])
        print(src.size(), trg[:, :-1].size(), predictions.size())
        loss = loss_function(predictions, trg_one_hot)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1
        del src, trg, trg_one_hot

    train_loss /= num_batches
    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)

    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            test_loss = 0
            num_batches = 0
            for data in dataset.test_loader:
                src = data['source'].to(model.device)
                trg = data['target'].to(model.device)
                trg_one_hot = torch.nn.functional.one_hot(trg, len(dataset.trg_vocab)).float()

                predictions = model(src, trg)
                loss = loss_function(predictions, trg_one_hot)

                test_loss += loss.item()
                num_batches += 1
                del src, trg, trg_one_hot

            test_loss /= num_batches
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()

save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
