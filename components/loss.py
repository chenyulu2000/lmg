import torch
from torch.nn.functional import cosine_embedding_loss

from components.similarity.pos_neg.pos_neg_sim import pos_neg_sim


def single_contrastive_loss(opt, im, s, s_l):
    scores = pos_neg_sim(
        opt=opt,
        images=im,
        captions=s,
        cap_lens=s_l
    )

    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.T.expand_as(scores)

    cost_s = (opt.margin + scores - d1).clamp(min=0)
    cost_im = (opt.margin + scores - d2).clamp(min=0)

    # clear diagonals
    regular = torch.eye(scores.size(0), device=scores.device)
    mask = regular.eq(1)
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    if opt.max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    loss = cost_s.sum() + cost_im.sum()
    return loss


def multi_contrastive_loss(opt, im, s, s_l, global_im=None, global_s=None):
    scores = pos_neg_sim(
        opt=opt,
        images=im,
        captions=s,
        cap_lens=s_l
    )

    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.T.expand_as(scores)

    cost_s = (opt.margin + scores - d1).clamp(min=0)
    cost_im = (opt.margin + scores - d2).clamp(min=0)

    # clear diagonals
    regular = torch.eye(scores.size(0), device=scores.device)
    mask = regular.eq(1)
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    if opt.max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    target = 2 * torch.empty(global_im.size(0), device=global_im.device).random_(2) - 1
    global_level_loss = cosine_embedding_loss(
        global_im,
        global_s,
        target
    )
    loss = cost_s.sum() + \
           cost_im.sum() + \
           global_im.size(0) * opt.global_weight * global_level_loss
    return loss
