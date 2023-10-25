import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, logging

from anatool import AnaArgParser, AnaLogger
from components.units import l2norm, use_cuda
from data import Vocabulary

logging.set_verbosity_error()


class BertEmb(nn.Module):
    def __init__(self, path, logger: AnaLogger):
        super(BertEmb, self).__init__()
        self.logger = logger
        self.bert = BertModel.from_pretrained(path)

    @property
    def embed_size(self):
        return 768

    def forward(self, x):
        bert_attention_mask = (x != 0).float()
        emb = self.bert(x, bert_attention_mask)[0]
        return emb


class GloveEmb(nn.Module):
    def __init__(self, num_embeddings, glove_dim, path, logger: AnaLogger):
        super(GloveEmb, self).__init__()

        self.logger = logger

        self.glove = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=glove_dim,
        )
        self.glove.weight = nn.Parameter(torch.load(f=path))
        self.glove_dim = glove_dim

    @property
    def embed_size(self):
        return self.glove_dim

    def forward(self, x):
        emb = self.glove(x)
        return emb


class BertTextEncoder(nn.Module):
    def __init__(self, opt, logger: AnaLogger):
        super(BertTextEncoder, self).__init__()
        self.latent_size = opt.embed_size
        self.no_txtnorm = opt.no_txtnorm
        self.logger = logger
        self.opt = opt

        self.embed = BertEmb(
            path=opt.bert_path,
            logger=logger
        )

        self.rnn = nn.LSTM(
            self.embed.embed_size,
            self.latent_size, opt.num_layers,
            batch_first=True,
            bidirectional=opt.use_bi_rnn
        )

    def forward(self, captions, lengths):
        caption_embed = self.embed(captions)

        packed = pack_padded_sequence(
            input=caption_embed,
            lengths=lengths.cpu(),
            batch_first=True
        )

        self.rnn.flatten_parameters()
        out, _ = self.rnn(packed)

        padded = pad_packed_sequence(
            sequence=out,
            batch_first=True,
            total_length=captions.size(1)
        )
        caption_embed, caption_len = padded

        caption_embed = (caption_embed[:, :, :caption_embed.size(2) // 2] +
                         caption_embed[:, :, caption_embed.size(2) // 2:]) / 2

        if not self.no_txtnorm:
            caption_embed = l2norm(tensor=caption_embed, dim=-1)
        if use_cuda(self.opt):
            caption_len = caption_len.cuda()
        return caption_embed, caption_len


class GloveTextEncoder(nn.Module):
    def __init__(self, opt, logger: AnaLogger, vocab: Vocabulary = None):
        super(GloveTextEncoder, self).__init__()
        self.latent_size = opt.embed_size
        self.no_txtnorm = opt.no_txtnorm
        self.logger = logger
        self.opt = opt

        self.embed = GloveEmb(
            num_embeddings=len(vocab),
            glove_dim=opt.word_dim,
            path=opt.glove_path,
            logger=logger
        )

        self.rnn = nn.LSTM(
            self.embed.embed_size,
            self.latent_size, opt.num_layers,
            batch_first=True,
            bidirectional=opt.use_bi_rnn
        )

    def forward(self, captions, lengths):
        caption_embed = self.embed(captions)

        packed = pack_padded_sequence(
            input=caption_embed,
            lengths=lengths.cpu(),
            batch_first=True
        )

        self.rnn.flatten_parameters()
        out, _ = self.rnn(packed)

        padded = pad_packed_sequence(
            sequence=out,
            batch_first=True,
            total_length=captions.size(1)
        )
        caption_embed, caption_len = padded

        caption_embed = (caption_embed[:, :, :caption_embed.size(2) // 2] +
                         caption_embed[:, :, caption_embed.size(2) // 2:]) / 2

        if not self.no_txtnorm:
            caption_embed = l2norm(tensor=caption_embed, dim=-1)
        if use_cuda(self.opt):
            caption_len = caption_len.cuda()
        return caption_embed, caption_len


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    vocab = Vocabulary(opt=opt, logger=logger)

    vocab.load(load_path=opt.vocab_word_idx_path)
    # te = TextEncoder(
    #     opt=opt,
    #     vocab=vocab,
    #     logger=logger
    # )
    # # tensor = torch.ones([32, 16], dtype=torch.int)
    # tensor = torch.IntTensor([[101, 2048, 102], [101, 102, 0], [101, 542, 102], [101, 102, 0]])
    # BE = BertEmb(logger=logger)
    # print(BE(tensor).shape)
