import torch
from interfaces import Module
from embedding import Embedding
from attention import MultiHeadAttention


class Transformer(Module):
    def __init__(self):
        super().__init__()

        test = MultiHeadAttention(512, 8, use_mask=True)
        result = test(torch.rand((64, 26, 512)))
        print(result.size())
        # print(result)


test = Transformer()
