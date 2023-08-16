import torch
import math
from torch import nn
from .interfaces import Module


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads=8, use_mask=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_mask = use_mask
        assert d_model % num_heads == 0, 'D_MODEL must be divisible by NUM_HEADS'

        # w_q_i projects D_MODEL to D_MODEL / NUM_HEADS. However, there are
        # NUM_HEADS parallel attention layers that are concatenated, so in the
        # end output dim is still D_MODEL / NUM_HEADS * NUM_HEADS = D_MODEL
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        # queries, keys, values = (64, seq, 512)
        # w_q = (512, 512)
        # queries @ w_q.t = (64, seq, 512)
        # split_heads = (64, 8, seq, 64)
        q = self.split_heads(self.w_q(queries))
        k = self.split_heads(self.w_k(keys))
        v = self.split_heads(self.w_v(values))

        # Perform NUM_HEADS parallel single-head attention
        attention = self.scaled_dot_product_attention(q, k, v)

        # Concatenate and return multi-headed results
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
        # This is essentially the inverse of split_heads
        batch_size, _, seq_len, _ = x.size()

        # Switch back to shape (batch, seq, num_heads, d_model)
        transposed = x.transpose(1, 2)

        # Merge last two dimensions
        return transposed.reshape(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, q, k, v):
        # Inputs are size (batch, num_heads, seq, d_model/num_heads)
        d_k = self.d_model // self.num_heads
        compatibility = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)

        """
        Use lower-triangular mask to prevent leftward information flow
        Fill lower triangle with negative infinity to zero out those values during softmax

        seq     weights      values          output
        0       [1 0 0]   [ --- a --- ]   [ a + 0 + 0 ]
        1       [1 1 0] * [ --- b --- ] = [ a + b + 0 ]
        2       [1 1 1]   [ --- c --- ]   [ a + b + c ]

        At seq=0, can only attend to seq=0
        At seq=1, can attend to both seq=0 and seq=1
        And so on...
        """
        if self.use_mask:
            seq_len = compatibility.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            compatibility += mask

        # Apply softmax along the last dimension
        value_weights = torch.softmax(compatibility, dim=-1)

        # Weight values by softmax results
        return torch.matmul(value_weights, v)
