import torch
import torch.nn as nn

from anatool import AnaLogger, AnaArgParser
from model.fusion.convolution.convolution import ConvolutionBlock
from model.fusion.matrix.matrix import MatrixBlock
from model.fusion.utils.layer_norm import LayerNorm
from model.fusion.attention.attention import MHAtt, AttentionFFN
from model.fusion.fourier.fourier import FNetBlock, FourierFFN
from model.fusion.utils.model_cfgs import FusionConfigs


class FourierSelfFusion(nn.Module):
    def __init__(self, using_ff_residual, logger: AnaLogger):
        super(FourierSelfFusion, self).__init__()
        fourier_config = FusionConfigs.FourierConfigs()
        self.using_ff_residual = using_ff_residual
        self.fn = FNetBlock()
        self.norm1 = LayerNorm(size=fourier_config.HIDDEN_SIZE)
        if using_ff_residual:
            self.ffn = FourierFFN(fourier_config=fourier_config, logger=logger)
            self.norm2 = LayerNorm(size=fourier_config.HIDDEN_SIZE)

    def forward(self, y):
        y = self.norm1(
            y + self.fn(y)
        )
        if self.using_ff_residual:
            y = self.norm2(
                y + self.ffn(y)
            )
        return y


class AttentionSelfFusion(nn.Module):
    def __init__(self, logger: AnaLogger):
        super(AttentionSelfFusion, self).__init__()
        attention_config = FusionConfigs.AttentionConfigs()

        self.logger = logger

        self.mh_att = MHAtt(attention_config=attention_config, logger=logger)
        self.ffn = AttentionFFN(attention_config=attention_config, logger=logger)

        self.dropout1 = nn.Dropout(p=attention_config.DROPOUT_R)
        self.norm1 = LayerNorm(size=attention_config.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(p=attention_config.DROPOUT_R)
        self.norm2 = LayerNorm(size=attention_config.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(
            y + self.dropout1(
                self.mh_att(v=y, k=y, q=y, mask=y_mask)
            )
        )
        y = self.norm2(
            y + self.dropout2(
                self.ffn(y)
            )
        )
        return y


class MatrixSelfFusion(nn.Module):
    def __init__(self, logger: AnaLogger):
        super(MatrixSelfFusion, self).__init__()
        matrix_config = FusionConfigs.MatrixConfigs()
        self.fn = MatrixBlock()
        self.norm = LayerNorm(size=matrix_config.HIDDEN_SIZE)
        self.logger = logger

    def forward(self, y):
        y = self.norm(
            y + self.fn(y)
        )
        return y


class ConvolutionSelfFusion(nn.Module):
    def __init__(self, logger: AnaLogger):
        super(ConvolutionSelfFusion, self).__init__()
        convolution_config = FusionConfigs.ConvolutionConfigs()
        self.fn = ConvolutionBlock()
        self.norm = LayerNorm(size=convolution_config.HIDDEN_SIZE)
        self.logger = logger

    def forward(self, y):
        y = self.norm(
            y + self.fn(y)
        )
        return y


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    # asf = AttentionSelfFusion(
    #     logger=logger
    # )
    i = torch.rand(32, 36, 1024)
    # mask = make_mask(i)
    # print(mask.size())
    # t = asf(y=i, y_mask=mask)
    # print(t.size())
    # fsf = FourierSelfFusion(
    #     logger=logger
    # )
    # t = fsf(y=i)
    # print(t.size())
