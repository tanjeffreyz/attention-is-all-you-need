from torch import nn
from interfaces import Module
from attention import MultiHeadAttention


class EncoderLayer(Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-headed attention and residual connection + layer norm
        attention_out = self.self_attention(queries=x, keys=x, values=x)
        x = self.layer_norm1(x + attention_out)

        # Feed-forward network and another residual + layer norm
        linear_out = self.linear(x)
        return self.layer_norm2(x + linear_out)
