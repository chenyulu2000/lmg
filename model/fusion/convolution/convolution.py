import torch.nn as nn

from anatool import AnaLogger
from model.fusion.convolution.fc import MLP
from model.fusion.utils.model_cfgs import FusionConfigs


class ConvolutionBlock(nn.Module):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).squeeze(1)
        return x


class ConvolutionFFN(nn.Module):
    def __init__(self, convolution_config: FusionConfigs.ConvolutionConfigs, logger: AnaLogger):
        super(ConvolutionFFN, self).__init__()

        self.logger = logger
        self.mlp = MLP(
            logger=logger,
            in_size=convolution_config.HIDDEN_SIZE,
            mid_size=convolution_config.FF_SIZE,
            out_size=convolution_config.HIDDEN_SIZE,
            dropout_r=convolution_config.DROPOUT_R,
            use_gelu=True
        )

    def forward(self, x):
        return self.mlp(x)
