from torch import nn
from interfaces import Module
from attention import MultiHeadAttention
from feed_forward import FeedForwardNetwork


class EncoderLayer(Module):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Multi-headed attention and residual connection + layer norm
        # Dropout is applied to sub-layer output, before residual and norm
        attention_out = self.self_attention(queries=x, keys=x, values=x)
        x = self.layer_norm1(x + self.dropout(attention_out))

        # Feed-forward network and another residual + layer norm
        ffn_out = self.ffn(x)
        return self.layer_norm2(x + self.dropout(ffn_out))
