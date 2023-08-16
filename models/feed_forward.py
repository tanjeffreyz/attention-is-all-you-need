import torch
from torch import nn
from interfaces import Module


class FeedForwardNetwork(Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
