import torch
import config
import os
import atexit
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


def append_loss(file_name, epoch, loss):
    with open(os.path.join(root, file_name), 'a') as file:
        file.write(f'{epoch}, {loss}\n')


# Save model whenever program terminates, just in case of crash
atexit.register(lambda: torch.save(
    model.state_dict(),
    os.path.join(weight_dir, f'{epoch:04}')
))

# Train
print()
for epoch in tqdm(range(config.NUM_EPOCHS), desc='Epoch'):
    model.train()
    train_loss = 0
    num_batches = 0     # Using DataPipe, cannot use len() to get number of batches
    for data in dataset.train_loader:
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)

        # Given the sequence length N, transformer tries to predict the N+1th token.
        # Thus, transformer must take in trg[:-1] as input and predict trg[1:] as output.
        optimizer.zero_grad()
        predictions = model(src, trg[:, :-1])

        # For CrossEntropyLoss, need to reshape input from (batch, seq_len, vocab_len)
        # to (batch * seq_len, vocab_len). Also need to reshape ground truth from
        # (batch, seq_len) to just (batch * seq_len)
        loss = loss_function(
            predictions.reshape(-1, predictions.size(-1)),
            trg[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1
        del src, trg

    train_loss /= num_batches
    writer.add_scalar('Loss/train', train_loss, epoch)
    append_loss('train.csv', epoch, train_loss)

    if epoch % 10 == 0:
        # Evaluate model
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            num_batches = 0
            for data in dataset.valid_loader:
                src = data['source'].to(model.device)
                trg = data['target'].to(model.device)

                predictions = model(src, trg[:, :-1])

                loss = loss_function(
                    predictions.reshape(-1, predictions.size(-1)),
                    trg[:, 1:].reshape(-1)
                )

                valid_loss += loss.item()
                num_batches += 1
                del src, trg

            valid_loss /= num_batches
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            append_loss('valid.csv', epoch, valid_loss)
