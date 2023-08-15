import config
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), config.D_MODEL)

    def forward(self, x):
        return self.embedding(x)
