import os
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from modules.embedding import Embedding


PATH = 'experiments/en-de/08_19_2023/17_27_32'
NUM_EPOCHS = 100


def plot_losses():
    train_df = pd.read_csv(
        os.path.join(PATH, 'scalars', 'loss', 'train.csv'),
        header=None
    ).head(NUM_EPOCHS)
    valid_df = pd.read_csv(
        os.path.join(PATH, 'scalars', 'loss', 'validation.csv'),
        header=None
    ).head(NUM_EPOCHS)

    sns.lineplot(train_df, x=0, y=1, label='Train Loss')
    ax = sns.lineplot(valid_df, x=0, y=1, label='Validation Loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Losses')
    plt.savefig(os.path.join(PATH, 'losses.png'))
    plt.show()


def plot_lr():
    lr_df = pd.read_csv(
        os.path.join(PATH, 'scalars', 'lr.csv'),
        header=None
    ).head(NUM_EPOCHS)

    ax = sns.lineplot(lr_df, x=0, y=1)
    ax.set(xlabel='Epoch', ylabel='Learning Rate',
           title='Learning Rate Schedule')
    plt.savefig(os.path.join(PATH, 'lr.png'))
    plt.show()


def plot_bleu():
    bleu_df = pd.read_csv(
        os.path.join(PATH, 'scalars', 'bleu.csv'),
        header=None
    ).head(NUM_EPOCHS)
    bleu_df[1] *= 100

    ax = sns.lineplot(bleu_df, x=0, y=1)
    ax.set(xlabel='Epoch', ylabel='BLEU Score',
           title='Validation BLEU Score')
    plt.savefig(os.path.join(PATH, 'bleu.png'))
    plt.show()


def plot_positional_encoding():
    embedding = Embedding(512, 1000, 0)
    data = torch.zeros((1, 50))
    pos_encoding = embedding.positional_encoding(data)

    ax = sns.heatmap(pos_encoding)
    ax.set(xlabel='Embedding Dimension', ylabel='Token Position', title='Positional Encoding')
    plt.savefig(os.path.join('docs', 'positional_encoding.png'))
    plt.show()


if __name__ == '__main__':
    plot_losses()
    plot_lr()
    plot_bleu()
    plot_positional_encoding()
