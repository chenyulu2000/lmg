import torch


def l1norm(tensor, dim, eps=1e-8):
    """L2-normalize columns of tensor."""
    norm = torch.abs(tensor).sum(dim=dim, keepdim=True) + eps
    tensor = torch.div(tensor, norm)
    return tensor


def l2norm(tensor, dim, eps=1e-8):
    """L2-normalize columns of tensor."""
    norm = torch.pow(tensor, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    tensor = torch.div(tensor, norm)
    return tensor


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / w2.clamp(min=eps)).squeeze()


def logsumexp(tensor, dim, coef):
    tensor = tensor.mul(coef).exp()
    tensor = tensor.sum(dim=dim, keepdim=True)
    tensor = tensor.log().div(coef)
    return tensor


def get_mask_attention(attn, batch_size, sourceL, queryL, coef=1):
    # attn shape: (bs, sourceL, queryL)
    mask_pos = attn.le(0)
    attn_pos = attn.masked_fill(mask_pos, torch.tensor(-1e9))
    attn_pos = torch.exp(attn_pos * coef)
    attn_pos = l1norm(tensor=attn_pos, dim=1)
    attn_pos = attn_pos.view(batch_size, queryL, sourceL)
    return attn_pos


def use_cuda(opt):
    if (opt.phase == 'train' and opt.local_rank > -1) or \
            (opt.phase == 'test' and len(opt.gpu_ids) > 0):
        return True
    return False


def use_global_optimizer(opt):
    if opt.multi_grain:
        return True
    return False
