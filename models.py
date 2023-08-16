import config
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), config.D_MODEL)

    @staticmethod
    def positional_encoding(x):
        result = torch.zeros((x.size(1), config.D_MODEL), dtype=torch.float)
        pos = torch.arange(0, x.size(1)).unsqueeze(1)
        dim = torch.arange(0, config.D_MODEL)

        # Sine for even positions, cosine for odd positions
        result[:, 0::2] = torch.sin(pos / (10_000 ** (dim[0::2] / config.D_MODEL)))
        result[:, 1::2] = torch.cos(pos / (10_000 ** (dim[1::2] / config.D_MODEL)))
        return result

    def forward(self, x):
        # Embedding shape: (batch, sequence_len, d_model)
        # Positional encoding shape: (sequence_len, d_model)
        return self.embedding(x) + self.positional_encoding(x)
