from torch import nn
from .interfaces import Module
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class DecoderLayer(Module):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads=num_heads, use_mask=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.enc_attention = MultiHeadAttention(d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # Multi-headed attention and residual connection + layer norm
        attention_out = self.self_attention(queries=x, keys=x, values=x)
        x = self.layer_norm1(x + self.dropout1(attention_out))

        # Multi-headed attention over output of encoder stack
        # Use ENC_OUT as the keys and values, the queries come from previous attention
        # Values come from encoder, so need to use encoder mask for this attention
        attention_out = self.enc_attention(queries=x, keys=enc_out, values=enc_out)
        x = self.layer_norm2(x + self.dropout2(attention_out))

        # Feed-forward network and another residual + layer norm
        ffn_out = self.ffn(x)
        return self.layer_norm3(x + self.dropout3(ffn_out))
