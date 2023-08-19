import torch
import config
import os
import seaborn as sns
from data import Dataset
from modules import Transformer
from matplotlib import pyplot as plt
from utils.experiment import Experiment


print('[~] Training')
print(f' ~  Using device: {Transformer.device}')

# Download and preprocess data
dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Initialize model
model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab),
    dataset.src_vocab[dataset.pad_token],
    dataset.trg_vocab[dataset.pad_token]
)

print(f' ~  Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# Set up experiment
experiment = Experiment(model, category='-'.join(config.LANGUAGE_PAIR))

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA1, config.BETA2),
    eps=config.EPS
)


# LR Scheduler
def get_lr(x):
    x += 1      # x is originally zero-indexed
    return (config.D_MODEL ** (-0.5)) * min(x ** (-0.5), x * (config.NUM_WARMUP ** (-1.5)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
ax = sns.lineplot(
    x=range(config.NUM_EPOCHS),
    y=[get_lr(x) for x in range(config.NUM_EPOCHS)]
)
ax.set(xlabel='Epoch', ylabel='Learning Rate', title='Learning Rate Schedule')
plt.savefig(os.path.join(experiment.path, 'lr_schedule.png'))

# Cross entropy loss
loss_function = torch.nn.CrossEntropyLoss()


# Train
def train(epoch):
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
        scheduler.step()

        train_loss += loss.item()
        num_batches += 1
        del src, trg

    train_loss /= num_batches
    experiment.add_scalar('loss/train', epoch, train_loss)

    if epoch % 10 == 0:
        validate(epoch)


# Evaluate against validation set and calculate BLEU
def validate(epoch):
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
        experiment.add_scalar('loss/validation', epoch, valid_loss)


experiment.loop(config.NUM_EPOCHS, train)
