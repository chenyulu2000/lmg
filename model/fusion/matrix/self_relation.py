import torch


def self_covariance(mat):
    numerator = torch.mm(mat, mat.T)
    no = torch.norm(mat, p=2, dim=1).unsqueeze(1)
    denominator = torch.mm(no, no.T)
    rel = numerator / (denominator + 1e-9)
    return rel


def self_relation(mat):
    mat_reducemean = mat - torch.mean(mat, dim=1, keepdim=True)
    numerator = torch.mm(mat_reducemean, mat_reducemean.T)
    no = torch.norm(mat_reducemean, p=2, dim=1).unsqueeze(1)
    denominator = torch.mm(no, no.T)
    corrcoef = numerator / (denominator + 1e-9)
    return corrcoef
