import torch
from torch import nn
from interfaces import Module
from embedding import Embedding
from encoder import EncoderLayer
from decoder import DecoderLayer


class Transformer(Module):
    def __init__(self,
                 d_model,
                 src_vocab_len,
                 trg_vocab_len,
                 num_heads=8,
                 num_layers=6,
                 dropout_rate=0.1):
        super().__init__()

        # Embeddings
        self.src_embedding = Embedding(src_vocab_len, d_model)
        self.trg_embedding = Embedding(trg_vocab_len, d_model)

        # Encoder
        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model,
                           num_heads=num_heads,
                           dropout_rate=dropout_rate)
              for _ in range(num_layers)]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Final linear layer to project embedding to target vocab word
        self.linear = nn.Linear(d_model, trg_vocab_len)

    def forward(self, source, target):
        # Encoder stack
        src_embedding = self.src_embedding(source)
        enc_out = self.encoder(src_embedding)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        # Final linear layer + softmax to get word probabilities
        return torch.softmax(self.linear(dec_out), dim=-1)
