from torch import nn
from interfaces import Module
from attention import MultiHeadAttention


class DecoderLayer(Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()

        self.d_model = d_model

        self.masked_attention = MultiHeadAttention(d_model, num_heads=num_heads, use_mask=True)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.enc_attention = MultiHeadAttention(d_model, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.linear = nn.Linear(self.d_model, self.d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # Multi-headed attention and residual connection + layer norm
        attention_out = self.masked_attention(queries=x, keys=x, values=x)
        x = self.layer_norm1(x + attention_out)

        # Multi-headed attention over output of encoder stack
        # Use ENC_OUT as the keys and values, the queries come from previous attention
        attention_out = self.enc_attention(queries=x, keys=enc_out, values=enc_out)
        x = self.layer_norm2(x + attention_out)

        # Feed-forward network and another residual + layer norm
        linear_out = self.linear(x)
        return self.layer_norm3(x + linear_out)
