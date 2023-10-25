import os

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from anatool import AnaLogger
from components import i2t_metrics, t2i_metrics
from components import use_cuda
from components.similarity.shard_xattn import shard_xattn


def validate(opt, model, summary_writer, global_iteration_step, dev_dataloader, logger: AnaLogger):
    model.eval()
    module = model.module if isinstance(model, DistributedDataParallel) else model
    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (_, _, lengths, _) in enumerate(dev_dataloader):
        max_n_word = max(max_n_word, max(lengths))

    for i, batch in enumerate(tqdm(dev_dataloader)):
        batch_img, batch_cap, batch_cap_len, batch_ids = batch
        if use_cuda(opt):
            batch_img = batch_img.cuda()
            batch_cap = batch_cap.cuda()
            batch_ids = batch_ids.cuda()
            batch_cap_len = batch_cap_len.cuda()
        with torch.no_grad():
            img, cap, cap_len, _, _ = module.forward(
                img=batch_img,
                cap=batch_cap,
                cap_len=batch_cap_len
            )
            if img_embs is None:
                dataset_len = len(dev_dataloader.dataset)
                img_embs = torch.zeros(dataset_len, img.size(1), img.size(2))
                cap_embs = torch.zeros(dataset_len, max_n_word, cap.size(2))
                cap_lens = [0] * dataset_len
                if use_cuda(opt):
                    img_embs = img_embs.cuda()
                    cap_embs = cap_embs.cuda()
            img_embs[batch_ids] = img
            cap_embs[batch_ids, :max(batch_cap_len), :] = cap
            for ii, nid in enumerate(batch_ids):
                cap_lens[nid] = cap_len[ii]

    img_embs = img_embs[::5, :, :]
    sims = shard_xattn(
        images=img_embs,
        captions=cap_embs,
        cap_lens=cap_lens,
        opt=opt,
        selected_thres_safe=opt.thres_safe,
        selected_using_intra_info=opt.using_intra_info
    )
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t_metrics(img_embs.cpu().numpy(), sims.cpu().numpy())
    metric_msg = 'Image to Text:\nr1: %.2f\nr5: %.2f\nr10: %.2f\nmedr: %.2f\nmeanr: %.2f\n' % (
        r1, r5, r10, medr, meanr)
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i_metrics(img_embs.cpu().numpy(), sims.cpu().numpy())
    metric_msg += '\nText to Image:\nr1: %.2f\nr5: %.2f\nr10: %.2f\nmedr: %.2f\nmeanr: %.2f\n' % (
        r1i, r5i, r10i, medri, meanri)
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    metric_msg += '\nrsum: %.2f\n' % currscore
    logger.info(metric_msg)

    # for observing rsum easily
    with open(os.path.join(opt.exp_dir, 'rsum.txt'), 'a') as f:
        f.write(f'{currscore}\n')
    with open(os.path.join(opt.exp_dir, 'I2T_rsum.txt'), 'a') as f:
        f.write('%.2f %.2f %.2f %.2f\n' % (r1, r5, r10, r1 + r5 + r10))
    with open(os.path.join(opt.exp_dir, 'T2I_rsum.txt'), 'a') as f:
        f.write('%.2f %.2f %.2f %.2f\n' % (r1i, r5i, r10i, r1i + r5i + r10i))
    with open(os.path.join(opt.exp_dir, 'thres_safe.txt'), 'a') as f:
        f.write('%.5f\n' % opt.thres_safe)
    caption_retrieval_metrics = {
        'R1': r1,
        'R5': r5,
        'R10': r10,
        'MEDR': medr,
        'MEANR': meanr
    }
    image_retrieval_metrics = {
        'R1': r1i,
        'R5': r5i,
        'R10': r10i,
        'MEDR': medri,
        'MEANR': meanri
    }
    summary_writer.add_scalars(
        main_tag='caption_retrieval_metrics',
        tag_scalar_dict=caption_retrieval_metrics,
        global_step=global_iteration_step
    )
    summary_writer.add_scalars(
        main_tag='image_retrieval_metrics',
        tag_scalar_dict=image_retrieval_metrics,
        global_step=global_iteration_step
    )
    summary_writer.add_scalar(
        tag='RSUM',
        scalar_value=currscore,
        global_step=global_iteration_step
    )

    return currscore
