import spacy
from torchtext.functional import to_tensor
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


# Patch Multi30K dataset, PyTorch's Google Drive link is no longer active
multi30k.URL['train'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz'
multi30k.URL['valid'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz'
multi30k.URL['test'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'


class Dataset:
    pipelines = {
        'en': spacy.load('en_core_web_sm'),
        'de': spacy.load('de_core_news_sm')
    }

    def __init__(self,
                 language_pair,
                 sos_token='<sos>',
                 eos_token='<eos>',
                 columns=('source', 'target'),
                 batch_size=64):
        self.src_lang, self.trg_lang = language_pair
        assert self.src_lang in Dataset.pipelines, 'Unrecognized source language'
        assert self.trg_lang in Dataset.pipelines, 'Unrecognized target language'

        self.sos_token = sos_token
        self.eos_token = eos_token

        # Load datasets
        train_data, _, test_data = Multi30k(root='data', language_pair=language_pair)

        # Build vocabs from dataset
        self.src_vocab = build_vocab_from_iterator(
            map(lambda x: x[0], train_data.map(self.tokenize)),
            specials=[self.sos_token, self.eos_token]
        )
        self.src_vocab.set_default_index(-1)

        self.trg_vocab = build_vocab_from_iterator(
            map(lambda x: x[1], train_data.map(self.tokenize)),
            specials=[self.sos_token, self.eos_token]
        )
        self.trg_vocab.set_default_index(-1)

        # Tokenize, encode, and batch the data
        train_processed = (
            train_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(batch_size)
            .rows2columnar(columns)
            .map(self.pad)
        )
        test_processed = (
            test_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(batch_size)
            .rows2columnar(columns)
            .map(self.pad)
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

        batch['source'] = to_tensor(batch['source'], padding_value=self.src_vocab[self.eos_token])
        batch['target'] = to_tensor(batch['target'], padding_value=self.trg_vocab[self.eos_token])
        return batch
