import datetime
import os.path

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from anatool import AnaLogger, AnaArgParser
from components import get_dataloader, use_cuda
from components.checkpointing import CheckpointManager
from components.solver import get_solver
from data import Vocabulary
from model.multi_grain_retrieval import MultiGrainRetrieval, SingleGrainRetrieval
from validate import validate


def train(opt, model, epoch, summary_writer, global_iteration_step, train_dataloader, optimizer_base, optimizer_global,
          running_loss, scheduler_base, scheduler_global, train_begin, logger: AnaLogger):
    model.train()
    module = model.module if isinstance(model, DistributedDataParallel) else model
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer_base.zero_grad()
        if optimizer_global is not None: optimizer_global.zero_grad()
        batch_img, batch_cap, batch_cap_len, _ = batch
        if use_cuda(opt):
            batch_img = batch_img.cuda()
            batch_cap = batch_cap.cuda()
            batch_cap_len = batch_cap_len.cuda()
        img, cap, cap_len, global_img, global_cap = module.forward(
            img=batch_img,
            cap=batch_cap,
            cap_len=batch_cap_len
        )
        batch_loss = module.loss(
            img=img,
            cap=cap,
            cap_len=cap_len,
            global_img=global_img,
            global_cap=global_cap
        )
        batch_loss.backward()
        if opt.grad_clip > 0:
            params = model.parameters()
            clip_grad_norm_(parameters=params, max_norm=opt.grad_clip)
        optimizer_base.step()
        if optimizer_global is not None: optimizer_global.step()

        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
        else:
            running_loss = batch_loss.item()

        scheduler_base.step()
        if scheduler_global is not None: scheduler_global.step()

        global_iteration_step += 1
        if global_iteration_step % 100 == 0:
            logger.info("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr_base: {:8f}][lr_global: {:8f}]".format(
                datetime.datetime.now() - train_begin, epoch,
                global_iteration_step, running_loss,
                optimizer_base.param_groups[0]['lr'],
                optimizer_global.param_groups[0]['lr'] if optimizer_global is not None else 0))

            summary_writer.add_scalar(
                tag='train/loss',
                scalar_value=batch_loss,
                global_step=global_iteration_step
            )
            summary_writer.add_scalar(
                tag='train/lr_base',
                scalar_value=optimizer_base.param_groups[0]['lr'],
                global_step=global_iteration_step
            )
            summary_writer.add_scalar(
                tag='train/lr_global',
                scalar_value=optimizer_global.param_groups[0]['lr'],
                global_step=global_iteration_step
            )
    torch.cuda.empty_cache()

    return running_loss


def main(opt, logger: AnaLogger):
    if opt.text_type == 'glove':
        vocab = Vocabulary(opt=opt, logger=logger)
        vocab.load(load_path=opt.vocab_word_idx_path)
    else:
        vocab = None

    if opt.multi_grain:
        model = MultiGrainRetrieval(
            opt=opt,
            logger=logger,
            vocab=vocab
        )
    else:
        model = SingleGrainRetrieval(
            opt=opt,
            logger=logger,
            vocab=vocab
        )
    if opt.local_rank != -1:
        model = DistributedDataParallel(
            module=model.cuda(),
            device_ids=[opt.local_rank]
        )

    if opt.local_rank < 1:
        total_param = sum([param.nelement() for param in model.parameters()])
        logger.info(f'Number of model parameter: %.2fM' % (total_param / 1e6))
        module = model.module if isinstance(model, DistributedDataParallel) else model

        if opt.multi_grain:
            fusion_param = sum([param.nelement() for param in module.cap_sf_list.parameters()] +
                               [param.nelement() for param in module.img_sf_list.parameters()]
                               )
            logger.info(f'Number of fusion parameter: %.2fK' % (fusion_param / 1e3))
    train_dataloader, train_dataset = get_dataloader(
        opt=opt,
        logger=logger,
        data_split='train',
        vocab=vocab,
    )
    dev_dataloader, dev_dataset = get_dataloader(
        opt=opt,
        logger=logger,
        data_split='dev',
        vocab=vocab,
    )

    optimizer_base, scheduler_base, optimizer_global, scheduler_global, iterations = get_solver(
        opt=opt,
        logger=logger,
        train_dataset=train_dataset,
        model=model
    )

    summary_writer = SummaryWriter(logdir=opt.exp_dir)
    checkpoint_manager = CheckpointManager(
        model=model,
        logger=logger,
        checkpoint_dirpath=os.path.join(opt.exp_dir, 'checkpoints')
    )

    global_iteration_step = 0

    running_loss = 0.0
    train_begin = datetime.datetime.now()
    best_rsum = 0.0

    for epoch in range(1, opt.num_epochs + 1):
        if opt.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch=epoch)

        logger.info(f'Training for epoch {epoch}.')
        running_loss = train(
            opt=opt,
            logger=logger,
            summary_writer=summary_writer,
            global_iteration_step=global_iteration_step,
            train_dataloader=train_dataloader,
            model=model,
            epoch=epoch,
            running_loss=running_loss,
            scheduler_base=scheduler_base,
            scheduler_global=scheduler_global,
            optimizer_base=optimizer_base,
            optimizer_global=optimizer_global,
            train_begin=train_begin
        )

        # only one process for validating
        if opt.local_rank > 0:
            continue
        logger.info(f'\nValidation after epoch {epoch}:')
        rsum = validate(
            opt=opt,
            logger=logger,
            summary_writer=None,
            global_iteration_step=global_iteration_step,
            dev_dataloader=dev_dataloader,
            model=model
        )

        # checkpoint_manager.step(epoch=epoch)

        if rsum > best_rsum:
            checkpoint_manager.save_best()
            logger.info(f'rsum: {best_rsum}  ----->  {rsum}.')
            logger.info(f'Save model at epoch {epoch}.')
            best_rsum = rsum
        else:
            logger.info(f'Not save model at epoch {epoch}.')


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    if opt.local_rank < 1:
        logger.info(opt)

    torch.manual_seed(opt.local_rank)
    torch.cuda.manual_seed_all(opt.local_rank)
    cudnn.benchmark = True
    cudnn.deterministic = True

    if use_cuda(opt):
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')

    logger.info(f'Running on: {opt.local_rank}.')
    main(opt=opt, logger=logger)

    if opt.local_rank < 1:
        logger.info(f'Training done! Time: {datetime.datetime.now()}.')
