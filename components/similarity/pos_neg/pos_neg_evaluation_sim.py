import torch

from components.similarity.relation import intra_relation, inter_relations_neg
from components.units import get_mask_attention, cosine_similarity


def pos_neg_evaluation_sim(opt, images, captions, cap_lens, selected_thres_safe, selected_using_intra_info):
    n_image, n_region = images.size(0), images.size(1)
    n_caption = captions.size(0)
    similarities = []

    contextT = torch.transpose(images, 1, 2)
    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        attn = torch.bmm(input=cap_i_expand, mat2=contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * selected_thres_safe

        # Neg-Pos branch
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        row_max = torch.max(attn_row, dim=1)[0].unsqueeze(-1)

        if selected_using_intra_info:
            attn_intra = intra_relation(K=cap_i, Q=cap_i, xlambda=5)
            attn_intra = attn_intra.repeat(batch_size, 1, 1)
            row_max_intra = torch.bmm(
                input=attn_intra,
                mat2=row_max
                .reshape(batch_size, n_word)
                .unsqueeze(-1)
            ).reshape(batch_size * n_word, -1)
            attn_neg = row_max_intra.lt(0).double()
        else:
            attn_neg = row_max.lt(0).float()
        t2i_sim_neg = row_max * attn_neg
        # negative effects
        t2i_sim_neg = t2i_sim_neg.view(batch_size, queryL)

        # positive attention
        # 1) positive effects based on aggregated features
        attn_pos = get_mask_attention(
            attn=attn_row,
            batch_size=batch_size,
            sourceL=sourceL,
            queryL=queryL,
            coef=opt.lambda_softmax
        )
        weightedContext_pos = torch.bmm(
            input=attn_pos,
            mat2=images
        )
        t2i_sim_pos_f = cosine_similarity(
            x1=cap_i_expand,
            x2=weightedContext_pos,
            dim=2
        )

        # 2) positive effects based on relevance scores
        attn_weight = inter_relations_neg(
            attn=attn_i,
            batch_size=batch_size,
            sourceL=n_region,
            queryL=n_word,
            xlambda=opt.lambda_softmax
        )
        # only for visualization
        # import numpy
        # model_name = 'LMG-C'
        # img_idx = i // 5
        # if i % 5 == 0:
        #     weight = attn_weight[img_idx]
        #     print(img_idx, weight.shape)
        #     numpy.save(f'vis/data/{model_name}/{img_idx}.npy', weight.cpu().numpy())

        t2i_sim_pos_r = attn.mul(attn_weight).sum(dim=-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r
        t2i_sim = t2i_sim_pos + t2i_sim_neg
        sim = t2i_sim.mean(dim=1, keepdim=True)

        similarities.append(sim)
    similarities = torch.cat(similarities, dim=1)
    return similarities
