import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_len, d_model):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, self.d_model)

    def positional_encoding(self, x):
        result = torch.zeros((x.size(1), self.d_model), dtype=torch.float)
        pos = torch.arange(0, x.size(1)).unsqueeze(1)
        dim = torch.arange(0, self.d_model)

        # Sine for even positions, cosine for odd positions
        result[:, 0::2] = torch.sin(pos / (10_000 ** (dim[0::2] / self.d_model)))
        result[:, 1::2] = torch.cos(pos / (10_000 ** (dim[1::2] / self.d_model)))
        return result

    def forward(self, x):
        # Embedding shape: (batch, sequence_len, d_model)
        # Positional encoding shape: (sequence_len, d_model)
        return self.embedding(x) + self.positional_encoding(x)
