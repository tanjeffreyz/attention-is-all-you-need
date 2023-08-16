import torch
from torch import nn
from .interfaces import Module


class Embedding(Module):
    def __init__(self, vocab_len, d_model, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Embedding shape: (batch, sequence_len, d_model)
        # Positional encoding shape: (sequence_len, d_model)
        return self.dropout(self.embedding(x) + self.positional_encoding(x))

    def positional_encoding(self, x):
        # result.shape = (seq_len, d_model)
        result = torch.zeros((x.size(1), self.d_model), dtype=torch.float)

        # pos.shape = (seq_len, 1)
        pos = torch.arange(0, x.size(1)).unsqueeze(1)

        # dim.shape = (d_model)
        dim = torch.arange(0, self.d_model)

        # Sine for even positions, cosine for odd positions
        result[:, 0::2] = torch.sin(pos / (10_000 ** (dim[0::2] / self.d_model)))
        result[:, 1::2] = torch.cos(pos / (10_000 ** (dim[1::2] / self.d_model)))
        return result
