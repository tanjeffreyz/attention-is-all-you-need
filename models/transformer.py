import torch
from torch import nn
from .interfaces import Module
from .embedding import Embedding
from .encoder import EncoderLayer
from .decoder import DecoderLayer


class Transformer(Module):
    def __init__(self,
                 d_model,
                 src_vocab_len,
                 trg_vocab_len,
                 src_pad_index,
                 trg_pad_index,
                 num_heads=8,
                 num_layers=6,
                 dropout_rate=0.1,
                 seed=20230815):
        super().__init__()

        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index

        # Manually seed to keep embeddings consistent across loads
        torch.manual_seed(seed)

        # Embeddings, pass in pad indices to prevent <pad> from contributing to gradient
        self.src_embedding = Embedding(d_model,
                                       src_vocab_len,
                                       src_pad_index,
                                       dropout_rate=dropout_rate)
        self.trg_embedding = Embedding(d_model,
                                       trg_vocab_len,
                                       trg_pad_index,
                                       dropout_rate=dropout_rate)

        # Encoder
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Decoder
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Final layer to project embedding to target vocab word probability distribution
        self.linear = nn.Linear(d_model, trg_vocab_len)

        # Move to GPU if possible
        self.to(self.device)

        # Re-seed afterward to allow shuffled data
        torch.seed()

    def forward(self, source, target):
        # Encoder stack
        enc_out = self.src_embedding(source)
        for layer in self.encoder_stack:
            enc_out = layer(enc_out)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_stack:
            dec_out = layer(dec_out, enc_out)

        # Final linear layer to get word probabilities
        # DO NOT apply softmax here, as CrossEntropyLoss already does normalization!!!
        return self.linear(dec_out)
