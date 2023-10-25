import torch
import torch.nn as nn

from anatool import AnaLogger
from model.fusion.fourier.fc import MLP
from model.fusion.utils.model_cfgs import FusionConfigs


class FNetBlock(nn.Module):
    def __init__(self):
        super(FNetBlock, self).__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FourierFFN(nn.Module):
    def __init__(self, fourier_config: FusionConfigs.FourierConfigs, logger: AnaLogger):
        super(FourierFFN, self).__init__()

        self.logger = logger
        self.mlp = MLP(
            logger=logger,
            in_size=fourier_config.HIDDEN_SIZE,
            mid_size=fourier_config.FF_SIZE,
            out_size=fourier_config.HIDDEN_SIZE,
            dropout_r=fourier_config.DROPOUT_R,
            use_gelu=True
        )

    def forward(self, x):
        return self.mlp(x)
