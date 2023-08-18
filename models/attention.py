import torch
import math
from torch import nn
from .interfaces import Module


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'D_MODEL must be divisible by NUM_HEADS'

        # w_q_i projects D_MODEL to D_MODEL / NUM_HEADS. However, there are
        # NUM_HEADS parallel attention layers that are concatenated, so in the
        # end output dim is still D_MODEL / NUM_HEADS * NUM_HEADS = D_MODEL
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        # queries, keys, values = (batch, seq, 512)
        # w_q = (512, 512)
        # queries @ w_q.t = (batch, seq, 512)
        # split_heads = (batch, 8, seq, 64)
        q = self.split_heads(self.w_q(queries))
        k = self.split_heads(self.w_k(keys))
        v = self.split_heads(self.w_v(values))

        # Perform NUM_HEADS parallel single-head attention
        attention = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # Concatenate and return multi-headed results
        # (batch, 8, seq, 64) -> (batch, seq, 512)
        merged = self.merge_heads(attention)

        # Apply final projection matrix
        return self.w_o(merged)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()

        # Split D_MODEL into NUM_HEADS channels of D_MODEL // NUM_HEADS each
        # Now shape is (batch, seq, num_heads, d_model/num_heads)
        heads = x.reshape(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads)

        # However, we want (batch, num_heads, seq, d_model/num_heads) because each tensor
        # of size (seq, d_model/num_heads) represents a single-head attention
        return heads.transpose(2, 1)

    def merge_heads(self, x):
        # Concatenate multi-headed results back into shape (batch, seq, d_model)
        # This is the inverse of split_heads
        batch_size, _, seq_len, _ = x.size()

        # Switch back to shape (batch, seq, num_heads, d_model)
        transposed = x.transpose(1, 2)

        # Merge last two dimensions
        return transposed.reshape(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # Inputs are size (batch, num_heads, seq, d_model/num_heads)
        d_k = self.d_model // self.num_heads
        compatibility = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)

        # Apply mask
        if mask is not None:
            compatibility = torch.masked_fill(compatibility, mask, float('-inf'))

        # Apply softmax along the last dimension
        value_weights = self.softmax(compatibility)

        # Weight values by softmax results
        return torch.matmul(value_weights, v)
