import torch
from interfaces import Module
from embedding import Embedding
from attention import MultiHeadAttention


class Transformer(Module):
    def __init__(self):
        super().__init__()

        test = MultiHeadAttention(512, 8, use_mask=True)
        x = torch.rand((64, 26, 512))
        result = test(x, x, x)
        print(result.size())
        # print(result)

    def forward(self, source, target):
        pass


test = Transformer()
