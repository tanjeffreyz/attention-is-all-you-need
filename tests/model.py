import unittest
import torch
from models.embedding import Embedding
from models.attention import MultiHeadAttention
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from models.feed_forward import FeedForwardNetwork
from models.transformer import Transformer


class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        tokens = torch.ones((64, 30), dtype=torch.int).to(Embedding.device)
        embedding = Embedding(512, 3000, 0).to(Embedding.device)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            embedding(tokens).size()
        )

    def test_positional_encoding(self):
        tokens = torch.ones((64, 30), dtype=torch.int).to(Embedding.device)
        embedding = Embedding(512, 3000, 0).to(Embedding.device)
        self.assertEqual(
            torch.Size([30, 512]),
            embedding.positional_encoding(tokens).size()
        )


class TestAttention(unittest.TestCase):
    def test_scaled_dot_product_attention(self):
        x = torch.rand((64, 8, 30, 64))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            torch.Size([64, 8, 30, 64]),
            attention.scaled_dot_product_attention(x, x, x).size()
        )

    def test_multi_head_attention(self):
        x = torch.rand((64, 30, 512))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            attention(x, x, x).size()
        )

    def test_split_heads(self):
        x = torch.rand((64, 30, 512))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            torch.Size([64, 8, 30, 64]),
            attention.split_heads(x).size()
        )

    def test_merge_heads(self):
        x = torch.rand((64, 8, 30, 64))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            attention.merge_heads(x).size()
        )


class TestTransformer(unittest.TestCase):
    def test_encoder_layer(self):
        x = torch.rand((64, 30, 512))
        encoder = EncoderLayer(512)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            encoder(x).size()
        )

    def test_decoder_layer(self):
        x = torch.rand((64, 30, 512)).to(DecoderLayer.device)
        decoder = DecoderLayer(512).to(DecoderLayer.device)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            decoder(x, x).size()
        )

    def test_feed_forward_network(self):
        x = torch.rand((64, 30, 512))
        ffn = FeedForwardNetwork(512)
        self.assertEqual(
            torch.Size([64, 30, 512]),
            ffn(x).size()
        )

    def test_transformer(self):
        x = torch.ones((64, 30), dtype=torch.int).to(Transformer.device)
        transformer = Transformer(512, 3000, 4000, src_pad_index=0, trg_pad_index=0)
        self.assertEqual(
            torch.Size([64, 30, 4000]),
            transformer(x, x).size()
        )


if __name__ == '__main__':
    unittest.main()
