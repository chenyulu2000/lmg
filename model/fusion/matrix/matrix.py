import torch
import torch.nn as nn

from anatool import AnaLogger
from model.fusion.matrix.fc import MLP
from model.fusion.matrix.self_relation import self_relation
from model.fusion.utils.model_cfgs import FusionConfigs


class MatrixBlock(nn.Module):
    def __init__(self):
        super(MatrixBlock, self).__init__()

    def forward(self, x):
        x_fusion = torch.zeros_like(x, device=x.device)
        for idx in range(x.size(0)):
            x_fusion[idx] = torch.mm(self_relation(x[idx]), x[idx])
        return x_fusion


class MatrixFFN(nn.Module):
    def __init__(self, matrix_config: FusionConfigs.MatrixConfigs, logger: AnaLogger):
        super(MatrixFFN, self).__init__()

        self.logger = logger
        self.mlp = MLP(
            logger=logger,
            in_size=matrix_config.HIDDEN_SIZE,
            mid_size=matrix_config.FF_SIZE,
            out_size=matrix_config.HIDDEN_SIZE,
            dropout_r=matrix_config.DROPOUT_R,
            use_gelu=True
        )

    def forward(self, x):
        return self.mlp(x)
