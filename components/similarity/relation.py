import torch
from torch import nn as nn

from components import l2norm


def intra_relation(K, Q, xlambda):
    """
    :param K: shape (n_context, sourceL, d)
    :param Q: shape (n_context, sourceL, d)
    :param xlambda:
    :return shape (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(input=Q, mat2=K)

    attn = attn.view(batch_size * KL, KL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


def inter_relations_neg(attn, batch_size, sourceL, queryL, xlambda):
    attn = nn.LeakyReLU(negative_slope=0.1)(attn)
    attn = l2norm(tensor=attn, dim=2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch * queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn
