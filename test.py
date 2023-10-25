import datetime
import os.path

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from tqdm import tqdm

from anatool import AnaArgParser, AnaLogger
from components import get_dataloader, use_cuda
from components.checkpointing import load_checkpoint
from components.metrics import i2t_metrics, t2i_metrics
from components.similarity.shard_xattn import shard_xattn
from data import Vocabulary
from model.multi_grain_retrieval import MultiGrainRetrieval, SingleGrainRetrieval


def test(opt, logger: AnaLogger, test_dataloader, model, selected_thres_safe, selected_using_intra_info, fold5=False):
    model.eval()
    module = model.module if isinstance(model, DataParallel) else model
    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0

    for i, (_, _, lengths, _) in enumerate(test_dataloader):
        max_n_word = max(max_n_word, max(lengths))
    for i, batch in enumerate(tqdm(test_dataloader)):
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
            # only for visualization
            # model_name = opt.
            # numpy.save('vis/output_features/{model_name}_I.npy', img.cpu().numpy())
            # numpy.save('vis/output_features/{model_name}_C.npy', cap.cpu().numpy())
            # numpy.save('vis/output_features/{model_name}_CL.npy', cap_len.cpu().numpy())
            # exit(0)

            if img_embs is None:
                dataset_len = len(test_dataloader.dataset)
                img_embs = torch.zeros(dataset_len, img.size(1), img.size(2), device=img.device)
                cap_embs = torch.zeros(dataset_len, max_n_word, cap.size(2), device=cap.device)
                cap_lens = [0] * dataset_len
            img_embs[batch_ids] = img
            cap_embs[batch_ids, :max(batch_cap_len), :] = cap
            for j, nid in enumerate(batch_ids):
                cap_lens[nid] = cap_len[j]

            if i == 0:
                torch.save(img, f=os.path.join(opt.exp_dir, 'first_batch_res_img'))
                torch.save(cap_embs, f=os.path.join(opt.exp_dir, 'first_batch_res_cap'))
                torch.save(cap_len, f=os.path.join(opt.exp_dir, 'first_batch_res_cap_len'))

    logger.info(f'Images: {img_embs.shape[0] / 5}, Captions: {cap_embs.shape[0]}.')

    rt, rti = None, None
    if not fold5:
        # no cross-validation, full evaluation
        img_embs = img_embs[::5, :, :]
        sims = shard_xattn(
            images=img_embs,
            captions=cap_embs,
            cap_lens=cap_lens,
            opt=opt,
            selected_thres_safe=selected_thres_safe,
            selected_using_intra_info=selected_using_intra_info
        )

        (r1, r5, r10, medr, meanr), rt = i2t_metrics(
            img_embs.cpu().numpy(),
            sims.cpu().numpy(),
            return_ranks=True
        )
        (r1i, r5i, r10i, medri, meanri), rti = t2i_metrics(
            img_embs.cpu().numpy(),
            sims.cpu().numpy(),
            return_ranks=True
        )
        ar = (r1 + r5 + r10) / 3
        ari = (r1i + r5i + r10i) / 3
        rsum = r1 + r5 + r10 + r1i + r5i + r10i
        msg = 'rsum: %.1f\n' % rsum
        msg += 'Average i2t Recall: %.1f\n' % ar
        msg += 'Image to text: %.1f %.1f %.1f %.1f %.1f\n' % (r1, r5, r10, medr, meanr)
        msg += 'Average t2i Recall: %.1f\n' % ari
        msg += 'Text to image: %.1f %.1f %.1f %.1f %.1f\n' % (r1i, r5i, r10i, medri, meanri)
        logger.info(msg)

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            sims = shard_xattn(
                images=img_embs_shard,
                captions=cap_embs_shard,
                cap_lens=cap_lens_shard,
                opt=opt,
                selected_thres_safe=selected_thres_safe,
                selected_using_intra_info=selected_using_intra_info
            )

            (r1, r5, r10, medr, meanr), rt0 = i2t_metrics(
                img_embs_shard.cpu().numpy(),
                sims.cpu().numpy(),
                return_ranks=True
            )
            logger.info(
                'Image to text: %.1f, %.1f, %.1f, %.1f, %.1f' %
                (r1, r5, r10, medr, meanr)
            )
            (r1i, r5i, r10i, medri, meanri), rti0 = t2i_metrics(
                img_embs_shard.cpu().numpy(),
                sims.cpu().numpy(),
                return_ranks=True
            )
            logger.info(
                'Text to image: %.1f, %.1f, %.1f, %.1f, %.1f' %
                (r1i, r5i, r10i, medri, meanri)
            )

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r1 + r5 + r10) / 3
            ari = (r1i + r5i + r10i) / 3
            rsum = r1 + r5 + r10 + r1i + r5i + r10i
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list((r1, r5, r10, medr, meanr)) + list((r1i, r5i, r10i, medri, meanri)) + [ar, ari, rsum]]

        msg = 'Mean metrics: \n'
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        msg += 'rsum: %.1f\n' % (mean_metrics[12])
        msg += 'Average i2t Recall: %.1f\n' % mean_metrics[10]
        msg += 'Image to text: %.1f %.1f %.1f %.1f %.1f\n' % mean_metrics[:5]
        msg += 'Average t2i Recall: %.1f\n' % mean_metrics[11]
        msg += 'Text to image: %.1f %.1f %.1f %.1f %.1f\n' % mean_metrics[5:10]
        logger.info(msg)

    torch.save({'rt': rt, 'rti': rti}, os.path.join(opt.exp_dir, 'ranks.pth.tar'))


def main(opt, logger: AnaLogger):
    if not os.path.exists(opt.load_path):
        logger.error(f'Checkpoint path {opt.load_path} does not exist.')
        raise FileNotFoundError
    vocab = Vocabulary(opt=opt, logger=logger)
    vocab.load(load_path=opt.vocab_word_idx_path)

    if opt.multi_grain:
        model = MultiGrainRetrieval(opt=opt, logger=logger, vocab=vocab)
    else:
        model = SingleGrainRetrieval(opt=opt, logger=logger, vocab=vocab)
    if use_cuda(opt):
        model = model.cuda()
    model = DataParallel(module=model, device_ids=opt.gpu_ids)
    model_state_dict = load_checkpoint(opt.load_path)

    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    logger.info(f'Loaded model from {opt.load_path}.')

    test_dataloader, _ = get_dataloader(
        opt=opt,
        logger=logger,
        data_split='test',
        vocab=vocab
    )

    test(
        opt=opt,
        logger=logger,
        test_dataloader=test_dataloader,
        model=model,
        fold5=opt.fold5,
        selected_thres_safe=opt.selected_thres_safe,
        selected_using_intra_info=opt.selected_using_intra_info
    )


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    logger.info(f'Starting the model run now: {datetime.datetime.now()}')
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True

    logger.info(f'Running on: {opt.gpu_ids}')
    if len(opt.gpu_ids) != 0:
        torch.cuda.set_device(device=torch.device('cuda', opt.gpu_ids[0]))

    logger.info('Starting testing.')
    main(opt=opt, logger=logger)
    logger.info(f'Test done! Time: {datetime.datetime.now()}')
