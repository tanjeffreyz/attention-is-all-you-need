import config
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

    def __init__(self, language_pair, columns=('source', 'target')):
        self.src_lang, self.trg_lang = language_pair
        assert self.src_lang in Dataset.pipelines, 'Unrecognized source language'
        assert self.trg_lang in Dataset.pipelines, 'Unrecognized target language'

        # Load datasets
        train_data, _, test_data = Multi30k(root='data', language_pair=config.LANGUAGE_PAIR)

        # Build vocabs from dataset
        self.src_vocab = build_vocab_from_iterator(
            map(lambda x: x[0], train_data.map(self.tokenize)),
            specials=[config.SOS, config.EOS]
        )
        self.src_vocab.set_default_index(-1)

        self.trg_vocab = build_vocab_from_iterator(
            map(lambda x: x[1], train_data.map(self.tokenize)),
            specials=[config.SOS, config.EOS]
        )
        self.trg_vocab.set_default_index(-1)

        # Tokenize, encode, and batch the data
        train_processed = (
            train_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(config.BATCH_SIZE)
            .rows2columnar(columns)
            .map(self.pad)
        )
        test_processed = (
            test_data
            .map(self.tokenize)
            .map(self.encode)
            .batch(config.BATCH_SIZE)
            .rows2columnar(columns)
            .map(self.pad)
        )
        self.train_loader = DataLoader(train_processed, batch_size=None, shuffle=True)
        self.test_loader = DataLoader(test_processed, batch_size=None, shuffle=False)

    def tokenize(self, pair):
        """Splits source and target sentences into tokens based on the configured language pair"""

        src, trg = pair
        src_tokens = [config.SOS] + [x.text.lower() for x in Dataset.pipelines[self.src_lang].tokenizer(src)] + [config.EOS]
        trg_tokens = [config.SOS] + [y.text.lower() for y in Dataset.pipelines[self.trg_lang].tokenizer(trg)] + [config.EOS]
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

        batch['source'] = to_tensor(batch['source'], padding_value=self.src_vocab[config.EOS])
        batch['target'] = to_tensor(batch['target'], padding_value=self.trg_vocab[config.EOS])
        return batch
