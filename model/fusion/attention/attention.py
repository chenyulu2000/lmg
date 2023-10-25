import math
import torch
import torch.nn as nn

from anatool import AnaLogger
from model.fusion.attention.fc import MLP
from model.fusion.utils.model_cfgs import FusionConfigs


# multi-head attention
class MHAtt(nn.Module):
    def __init__(self, attention_config: FusionConfigs.AttentionConfigs, logger: AnaLogger):
        super(MHAtt, self).__init__()
        self.logger = logger
        self.attention_config = attention_config

        self.linear_v = nn.Linear(
            in_features=attention_config.HIDDEN_SIZE,
            out_features=attention_config.HIDDEN_SIZE
        )
        self.linear_k = nn.Linear(
            in_features=attention_config.HIDDEN_SIZE,
            out_features=attention_config.HIDDEN_SIZE
        )
        self.linear_q = nn.Linear(
            in_features=attention_config.HIDDEN_SIZE,
            out_features=attention_config.HIDDEN_SIZE
        )
        self.linear_merge = nn.Linear(
            in_features=attention_config.HIDDEN_SIZE,
            out_features=attention_config.HIDDEN_SIZE
        )
        self.dropout = nn.Dropout(p=attention_config.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        # v.size() (bs, proposal/len, emb_size)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.attention_config.MULTI_HEAD,
            int(self.attention_config.HIDDEN_SIZE / self.attention_config.MULTI_HEAD)
        ).transpose(1, 2)
        # v.size() (bs, heads, proposal/len, emb_size)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.attention_config.MULTI_HEAD,
            int(self.attention_config.HIDDEN_SIZE / self.attention_config.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.attention_config.MULTI_HEAD,
            int(self.attention_config.HIDDEN_SIZE / self.attention_config.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        # self.logger.debug(atted.size())

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.attention_config.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        # score.size() (bs*rounds, heads, proposal/len, proposal/len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # self.logger.debug(scores.size())

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = nn.functional.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        # self.logger.debug(att_map.size())

        return torch.matmul(att_map, value)


# feed forward net
class AttentionFFN(nn.Module):
    def __init__(self, attention_config: FusionConfigs.AttentionConfigs, logger: AnaLogger):
        super(AttentionFFN, self).__init__()

        self.logger = logger
        self.mlp = MLP(
            logger=logger,
            in_size=attention_config.HIDDEN_SIZE,
            mid_size=attention_config.FF_SIZE,
            out_size=attention_config.HIDDEN_SIZE,
            dropout_r=attention_config.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)