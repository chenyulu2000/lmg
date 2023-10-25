import torch.nn as nn

from anatool import AnaLogger


# full-connected layer
class FC(nn.Module):
    def __init__(self, logger: AnaLogger, in_size, out_size, dropout_r=0., use_gelu=True):
        super(FC, self).__init__()
        self.logger = logger
        self.dropout_r = dropout_r
        self.use_gelu = use_gelu

        self.linear = nn.Linear(in_features=in_size, out_features=out_size)

        if use_gelu:
            self.gelu = nn.GELU()

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_gelu:
            x = self.gelu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


# multi-layer perception
class MLP(nn.Module):
    def __init__(self, logger: AnaLogger, in_size, mid_size, out_size, dropout_r=0., use_gelu=True):
        super(MLP, self).__init__()

        self.fc = FC(
            logger=logger,
            in_size=in_size,
            out_size=mid_size,
            dropout_r=dropout_r,
            use_gelu=use_gelu
        )
        self.linear = nn.Linear(in_features=mid_size, out_features=out_size)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        return self.dropout(self.linear(self.fc(x)))
