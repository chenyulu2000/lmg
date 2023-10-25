import random

import nltk.tokenize
import torch
from torch.utils.data import Dataset
from anatool import AnaLogger, AnaArgParser
from data.vocabulary import Vocabulary
import numpy as np
from transformers import BertTokenizer


class BertImageCaptionDataset(Dataset):
    def __init__(self, opt, data_split, logger: AnaLogger):
        self.logger = logger
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        self.data_split = data_split

        if opt.is_debug and data_split == 'train':
            data_split = 'dev'

        if data_split == 'train':
            image_features_path = opt.train_image_path
            cap_path = opt.train_cap_path
        elif data_split == 'dev':
            image_features_path = opt.dev_image_path
            cap_path = opt.dev_cap_path
        elif data_split == 'test':
            image_features_path = opt.test_image_path
            cap_path = opt.test_cap_path
        else:
            logger.error(f'Invalid data split: {data_split}.')
            raise ValueError

        self.caption_non = []
        try:
            with open(cap_path, 'r') as f:
                for line in f:
                    self.caption_non.append(line.strip())
        except Exception:
            logger.error(f'Invalid caption path: {cap_path}.')
        self.length = len(self.caption_non)

        self.image_features = np.load(image_features_path, allow_pickle=True)
        if opt.local_rank == 0 or opt.local_rank == -1:
            logger.info(f'{str(data_split).upper()} Image shape: {self.image_features.shape}.')
            logger.info(f'{str(data_split).upper()} Text shape: {len(self.caption_non)}.')

        if self.image_features.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # coco dev size 5070 -> 5000
        if data_split == 'dev':
            self.length = 5000

        # for visualization
        # if data_split == 'test':
        #     self.length = 500

    def __getitem__(self, index):
        img_id = index // self.im_div
        image = torch.Tensor(self.image_features[img_id])
        caption_non = self.caption_non[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption_non)
        output_tokens = []
        deleted_idx = []
        for i, token in enumerate(caption_tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(
                text=str(token).lower()
            )
            prob = random.random()

            if prob < 0.2 and self.data_split == 'train':
                prob /= 0.2

                if prob < 0.5:
                    for _ in sub_tokens:
                        output_tokens.append('[MASK]')
                elif prob < 0.6:
                    for _ in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        deleted_idx.append(len(output_tokens) - 1)
            else:
                output_tokens.extend(sub_tokens)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
        caption = self.tokenizer.convert_tokens_to_ids(
            tokens=['[CLS]'] + output_tokens + ['[SEP]']
        )
        caption = torch.Tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length


class GloveImageCaptionDataset(Dataset):
    def __init__(self, opt, data_split, vocab: Vocabulary, logger: AnaLogger):
        self.vocab = vocab
        self.logger = logger
        self.opt = opt

        if opt.is_debug and data_split == 'train':
            # use dev for debugging
            data_split = 'dev'

        if data_split == 'train':
            image_features_path = opt.train_image_path
            cap_path = opt.train_cap_path
        elif data_split == 'dev':
            image_features_path = opt.dev_image_path
            cap_path = opt.dev_cap_path
        elif data_split == 'test':
            image_features_path = opt.test_image_path
            cap_path = opt.test_cap_path
        else:
            logger.error(f'Invalid data split: {data_split}.')
            raise ValueError

        self.caption_non = []
        try:
            with open(cap_path, 'r') as f:
                for line in f:
                    self.caption_non.append(line.strip())
        except Exception:
            logger.error(f'Invalid caption path: {cap_path}.')
        self.length = len(self.caption_non)

        self.image_features = np.load(image_features_path, allow_pickle=True)
        if opt.local_rank == 0 or opt.local_rank == -1:
            logger.info(f'{str(data_split).upper()} Image shape: {self.image_features.shape}.')
            logger.info(f'{str(data_split).upper()} Text shape: {len(self.caption_non)}.')

        if self.image_features.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # coco dev size 5070 -> 5000
        if data_split == 'dev':
            self.length = 5000

        # for visualization
        # if data_split == 'test':
        #     self.length = 500

    def __getitem__(self, index):
        img_id = index // self.im_div
        image = torch.Tensor(self.image_features[img_id])
        caption_non = self.caption_non[index]
        vocab = self.vocab

        tokens = nltk.tokenize.word_tokenize(
            text=str(caption_non).lower()
        )
        caption_non = [vocab('<start>')] + [
            vocab(token)
            for token in tokens
        ] + [vocab('<end>')]

        caption = torch.Tensor(caption_non)
        return image, caption, index, img_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    vocab = Vocabulary(opt=opt, logger=logger)

    vocab.load(load_path=opt.vocab_word_idx_path)

    test_dataset = GloveImageCaptionDataset(
        opt=opt,
        data_split='test',
        vocab=vocab,
        logger=logger
    )
