import torch.nn as nn

from anatool import AnaLogger, AnaArgParser
from components import multi_contrastive_loss, single_contrastive_loss
from data import Vocabulary
from model.encoders.image_encoder import ImageEncoder
from model.encoders.text_encoder import GloveTextEncoder, BertTextEncoder
from model.fusion.utils.make_mask import make_mask
from model.fusion.self_fusion import FourierSelfFusion, AttentionSelfFusion, MatrixSelfFusion, ConvolutionSelfFusion


class SingleGrainRetrieval(nn.Module):
    def __init__(self, opt, vocab: Vocabulary, logger: AnaLogger):
        super(SingleGrainRetrieval, self).__init__()
        self.opt = opt

        self.logger = logger
        self.img_encoder = ImageEncoder(opt=opt, logger=logger)
        if opt.text_type == 'glove':
            self.txt_encoder = GloveTextEncoder(opt=opt, vocab=vocab, logger=logger)
        else:
            self.txt_encoder = BertTextEncoder(opt=opt, logger=logger)

    def forward(self, img, cap, cap_len):
        img = self.img_encoder(img)
        cap, cap_len = self.txt_encoder(cap, cap_len)
        return img, cap, cap_len, None, None

    def loss(self, img, cap, cap_len, **kwargs):
        loss = single_contrastive_loss(
            im=img,
            s=cap,
            s_l=cap_len,
            opt=self.opt
        )
        return loss


class MultiGrainRetrieval(nn.Module):
    def __init__(self, opt, vocab: Vocabulary, logger: AnaLogger):
        super(MultiGrainRetrieval, self).__init__()
        self.opt = opt

        self.logger = logger
        self.img_encoder = ImageEncoder(opt=opt, logger=logger)
        if opt.text_type == 'glove':
            self.txt_encoder = GloveTextEncoder(opt=opt, vocab=vocab, logger=logger)
        else:
            self.txt_encoder = BertTextEncoder(opt=opt, logger=logger)

        self.sf_stack = opt.sf_stack

        self.pool = nn.AdaptiveAvgPool2d((36, opt.embed_size))

        if len(opt.sf_stack) != \
                opt.sf_stack.count('F') + \
                opt.sf_stack.count('A') + \
                opt.sf_stack.count('M') + \
                opt.sf_stack.count('C'):
            logger.error(f'Invalid SF stack {opt.sf_stack}.')
            raise ValueError

        self.img_sf_list = nn.ModuleList()
        self.cap_sf_list = nn.ModuleList()
        for sf_type in opt.sf_stack:
            if sf_type == 'A':
                self.img_sf_list.append(AttentionSelfFusion(logger=logger))
                self.cap_sf_list.append(AttentionSelfFusion(logger=logger))
            elif sf_type == 'F':
                self.img_sf_list.append(FourierSelfFusion(logger=logger, using_ff_residual=opt.using_ff_residual))
                self.cap_sf_list.append(FourierSelfFusion(logger=logger, using_ff_residual=opt.using_ff_residual))
            elif sf_type == 'M':
                self.img_sf_list.append(MatrixSelfFusion(logger=logger))
                self.cap_sf_list.append(MatrixSelfFusion(logger=logger))
            else:
                self.img_sf_list.append(ConvolutionSelfFusion(logger=logger))
                self.cap_sf_list.append(ConvolutionSelfFusion(logger=logger))

    def forward(self, img, cap, cap_len):
        img = self.img_encoder(img)
        cap, cap_len = self.txt_encoder(cap, cap_len)
        global_img = img
        global_cap = cap
        img_mask = make_mask(feature=global_img)
        cap_mask = make_mask(feature=global_cap)
        for idx, sf_type in enumerate(self.sf_stack):
            img_sf = self.img_sf_list[idx]
            cap_sf = self.cap_sf_list[idx]
            if sf_type == 'A':
                global_img = img_sf(global_img, img_mask)
                global_cap = cap_sf(global_cap, cap_mask)
            else:
                global_img = img_sf(global_img)
                global_cap = cap_sf(global_cap)

        global_cap = self.pool(global_cap)
        global_img = global_img.reshape(global_img.size(0), -1)
        global_cap = global_cap.reshape(global_cap.size(0), -1)

        return img, cap, cap_len, global_img, global_cap

    def loss(self, img, cap, cap_len, global_img, global_cap):
        loss = multi_contrastive_loss(
            im=img,
            s=cap,
            s_l=cap_len,
            opt=self.opt,
            global_im=global_img,
            global_s=global_cap
        )
        return loss


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
