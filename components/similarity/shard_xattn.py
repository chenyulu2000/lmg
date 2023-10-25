import torch

from components.similarity.pos_neg.pos_neg_evaluation_sim import pos_neg_evaluation_sim


def shard_xattn(images, captions, cap_lens, opt, selected_thres_safe, selected_using_intra_info,
                shard_size=500):
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = torch.zeros(len(images), len(captions), device=images.device)
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = images[im_start:im_end]
            s = captions[cap_start:cap_end]
            l = cap_lens[cap_start:cap_end]
            sim = pos_neg_evaluation_sim(
                opt=opt,
                images=im,
                captions=s,
                cap_lens=l,
                selected_thres_safe=selected_thres_safe,
                selected_using_intra_info=selected_using_intra_info
            )
            d[im_start:im_end, cap_start:cap_end] = sim
    return d
