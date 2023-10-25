"""
A Vocabulary maintains a mapping between words and corresponding unique
integers, holds special integers (tokens) for indicating start and end of
sequence, and offers functionality to map out-of-vocabulary words to the
corresponding token.
"""
import json
from collections import Counter
from typing import List

import nltk
from tqdm import tqdm

from anatool import AnaLogger, AnaArgParser


class Vocabulary:
    """
    A simple Vocabulary class which maintains a mapping between words and
    integer tokens.
    """

    def __init__(self, opt, logger: AnaLogger, load: bool = False):
        self.opt = opt
        self.logger = logger

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        if load == True:
            self.load(load_path=opt.vocab_word_idx_path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, save_path):
        try:
            with open(save_path, 'w') as f:
                word_idx_dict = {
                    'word2idx': self.word2idx,
                    'idx2word': self.idx2word,
                    'idx': self.idx
                }
                json.dump(word_idx_dict, f)
        except Exception:
            self.logger.error(f'Invalid vocab path: {save_path}.')

    def load(self, load_path):
        try:
            with open(load_path, 'r') as f:
                word_idx_dict = json.load(f)

                self.word2idx = word_idx_dict['word2idx']
                self.idx2word = {
                    v: k
                    for k, v in self.word2idx.items()
                }
                self.idx = max(self.idx2word)
        except Exception:
            self.logger.error(f'Invalid vocab path: {load_path}.')

    def build_vocab_from_origin_text(self, origin_text_path_list: List[str], threshold):
        counter = Counter()
        captions = []
        try:
            for path in origin_text_path_list:
                with open(path, 'rb') as f:
                    for line in f:
                        captions.append(line.strip())
        except Exception:
            self.logger.error(f'Invalid text path list: {origin_text_path_list}.')
        for i, caption in tqdm(enumerate(captions)):
            tokens = nltk.tokenize.word_tokenize(
                text=caption.lower().decode('utf-8')
            )
            counter.update(tokens)
        self.logger.info(f'Tokenized captions length: {len(captions)}')

        # Discard words that appear less often.
        words = [
            word
            for word, cnt in counter.items()
            if cnt >= threshold
        ]

        self.add_word(word='<pad>')
        self.add_word(word='<start>')
        self.add_word(word='<end>')
        self.add_word(word='<unk>')

        for word in words:
            self.add_word(word=word)


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    vocab = Vocabulary(opt=opt, logger=logger)

    vocab.build_vocab_from_origin_text(
        origin_text_path_list=[
            opt.train_cap_path,
            opt.dev_cap_path,
            # opt.test_cap_path
        ],
        threshold=0
    )

    vocab.save(save_path=opt.vocab_word_idx_path)
