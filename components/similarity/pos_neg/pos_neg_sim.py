import torch

from components import use_cuda
from components.units import get_mask_attention, cosine_similarity
from ..relation import inter_relations_neg


def pos_neg_sim(opt, images, captions, cap_lens):
    n_image, n_region = images.size(0), images.size(1)
    n_caption = captions.size(0)
    cap_len_i = torch.zeros(1, n_caption)
    if use_cuda(opt):
        cap_len_i = cap_len_i.cuda()
    similarities = []

    max_pos = []
    max_neg = []

    N_POS_WORD = 0

    for i in range(n_caption):
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        cap_len_i[0, i] = n_word
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        contextT = torch.transpose(images, 1, 2)
        attn = torch.bmm(input=cap_i_expand, mat2=contextT)
        attn_i = torch.transpose(attn, 1, 2).contiguous()
        attn_thres = attn - torch.ones_like(attn) * opt.thres

        # Neg-Pos branch
        batch_size, queryL, sourceL = images.size(0), cap_i_expand.size(1), images.size(1)
        attn_row = attn_thres.view(batch_size * queryL, sourceL)
        row_max = torch.max(attn_row, dim=1)[0].unsqueeze(-1)
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
        weightedContext_pos = torch.bmm(input=attn_pos, mat2=images)
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
        t2i_sim_pos_r = attn.mul(attn_weight).sum(dim=-1)

        t2i_sim_pos = t2i_sim_pos_f + t2i_sim_pos_r
        t2i_sim = t2i_sim_pos + t2i_sim_neg
        sim = t2i_sim.mean(dim=1, keepdim=True)

        wrong_index = sim.sort(0, descending=True)[1][0].item()
        if wrong_index == i:
            attn_max_row = torch.max(attn.reshape(batch_size * n_word, n_region).squeeze(), dim=1)[0]
            attn_max_row_pos = attn_max_row[(i * n_word):(i * n_word + n_word)]

            # negative samples
            neg_index = sim.sort(0)[1][0].item()
            attn_max_row_neg = attn_max_row[(neg_index * n_word):(neg_index * n_word + n_word)]

            max_pos.append(attn_max_row_pos)
            max_neg.append(attn_max_row_neg)
            N_POS_WORD += n_word

            if N_POS_WORD > 200:
                max_pos_aggre = torch.cat(max_pos, dim=0)
                max_neg_aggre = torch.cat(max_neg, dim=0)
                mean_pos = max_pos_aggre.mean()
                mean_neg = max_neg_aggre.mean()
                stnd_pos = max_pos_aggre.std()
                stnd_neg = max_neg_aggre.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)
                B = 2 * (mean_pos * stnd_neg.pow(2) - mean_neg * stnd_pos.pow(2))
                C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + \
                    2 * (stnd_pos * stnd_neg).pow(2) * (stnd_neg / (opt.alpha * stnd_pos) + 1e-8).log()

                thres = opt.thres
                thres_safe = opt.thres_safe

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    opt.thres = ((-B + E.sqrt()) / (2 * A + 1e-10)).item()
                    opt.thres_safe = (mean_pos - 3 * stnd_pos).item()

                opt.thres = max(0, min(opt.thres, 1))
                opt.thres_safe = max(0, min(opt.thres_safe, 1))

                opt.thres = 0.7 * opt.thres + 0.3 * thres
                opt.thres_safe = 0.7 * opt.thres_safe + 0.3 * thres_safe

        if N_POS_WORD < 200:
            opt.thres = 0
            opt.thres_safe = 0

        similarities.append(sim)
    similarities = torch.cat(similarities, dim=1)
    return similarities
