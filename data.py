import torch
import spacy
from torchtext.functional import to_tensor
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


# Patch Multi30K dataset, PyTorch's Google Drive link is no longer active
multi30k.URL['train'] = 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/training.tar.gz'
multi30k.URL['valid'] = 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/validation.tar.gz'
multi30k.URL['test'] = 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/mmt16_task1_test.tar.gz'
multi30k.MD5['test'] = 'd914ec964e2c5f0534e5cdd3926cd2fe628d591dad9423c3ae953d93efdb27a6'


class Dataset:
    pipelines = {
        'en': spacy.load('en_core_web_sm'),
        'de': spacy.load('de_core_news_sm')
    }

    def __init__(self,
                 language_pair,
                 sos_token='<sos>',
                 eos_token='<eos>',
                 unk_token='<unk>',
                 pad_token='<pad>',
                 columns=('source', 'target'),
                 batch_size=64):
        self.src_lang, self.trg_lang = language_pair
        assert self.src_lang in Dataset.pipelines, 'Unrecognized source language'
        assert self.trg_lang in Dataset.pipelines, 'Unrecognized target language'

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        # Load datasets
        train_data, _, test_data = Multi30k(root='data', language_pair=language_pair)

        # Build vocabs from dataset
        self.src_vocab = build_vocab_from_iterator(
            map(lambda x: x[0], train_data.map(self.tokenize)),
            specials=[self.sos_token, self.eos_token, self.unk_token, self.pad_token]
        )
        self.src_vocab.set_default_index(self.src_vocab[self.unk_token])

        self.trg_vocab = build_vocab_from_iterator(
            map(lambda x: x[1], train_data.map(self.tokenize)),
            specials=[self.sos_token, self.eos_token, self.unk_token, self.pad_token]
        )
        self.trg_vocab.set_default_index(self.trg_vocab[self.unk_token])

        # Tokenize, encode, and batch the data
        train_processed = (
            train_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(batch_size)
            .rows2columnar(columns)
            .map(self.pad)
            .map(self.one_hot)
        )
        test_processed = (
            test_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(batch_size)
            .rows2columnar(columns)
            .map(self.pad)
            .map(self.one_hot)
        )
        self.train_loader = DataLoader(train_processed, batch_size=None, shuffle=True)
        self.test_loader = DataLoader(test_processed, batch_size=None, shuffle=False)

    def tokenize(self, pair):
        """Splits source and target sentences into tokens based on the configured language pair"""

        src, trg = pair
        src_tokens = [self.sos_token] + [x.text.lower() for x in Dataset.pipelines[self.src_lang].tokenizer(src)] + [self.eos_token]
        trg_tokens = [self.sos_token] + [y.text.lower() for y in Dataset.pipelines[self.trg_lang].tokenizer(trg)] + [self.eos_token]
        return src_tokens, trg_tokens

    def encode(self, pair):
        """Replaces tokens with their corresponding vocabulary indices"""

        src, trg = pair
        return self.src_vocab.lookup_indices(src), self.trg_vocab.lookup_indices(trg)

    def pad(self, batch):
        """
        Pads all sequences in the batch to have same length. Pads using the EOS token because if <eos> is reached, model
        should insist sequence has finished and output <eos> even if prompted further.
        """

        # Pad source and target together so that all sequence lengths are equal!!!
        zipped = to_tensor(
            batch['source'] + batch['target'],
            padding_value=self.src_vocab[self.pad_token]
        )

        # Separate source and target sequences again
        mid = zipped.size(0) // 2
        batch['source'] = zipped[:mid]
        batch['target'] = zipped[mid:]
        return batch

    def one_hot(self, batch):
        """
        One-hot encodes all target sequences for use in the loss function.
        This is necessary because Transformer predicts a probability distribution.
        """

        batch['target.one_hot'] = torch.nn.functional.one_hot(batch['target'], len(self.trg_vocab)).float()
        return batch
