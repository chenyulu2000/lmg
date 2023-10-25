import torch
import torch.nn as nn

from components.units import l2norm
from anatool import AnaLogger, AnaArgParser
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(self, opt, logger: AnaLogger):
        super(ImageEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.no_imgnorm = opt.no_imgnorm
        self.logger = logger

        self.linear = nn.Linear(
            in_features=opt.img_dim,
            out_features=self.embed_size
        )

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        # self.init_weights()

    # def init_weights(self):
    #     r = np.sqrt(6.) / np.sqrt(
    #         self.linear.in_features +
    #         self.linear.out_features
    #     )
    #     self.linear.weight.data.uniform_(-r, r)
    #     self.linear.bias.data.fill_(0)

    def forward(self, images):
        """
        :param images: shape (bs, 36, img_dim)
        :return: shape (bs, 36, embed_dim)
        """
        # print(images)
        features = self.linear(images)
        if not self.no_imgnorm:
            features = l2norm(
                tensor=features,
                dim=-1,
            )
        return features


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)

    ie = ImageEncoder(
        opt=opt,
        logger=logger
    )
    print(ie(torch.rand(32, 36, 2048)).shape)
