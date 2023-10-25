import torch.cuda
from math import sqrt
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler

from anatool import AnaLogger
from components.units import use_global_optimizer


def get_solver(opt, logger: AnaLogger, model, train_dataset):
    initial_lr = opt.learning_rate
    if torch.cuda.device_count() > 0:
        nproc = torch.cuda.device_count()
    else:
        nproc = 1
    iterations = 1 + len(train_dataset) // (opt.train_batch_size * nproc)

    def lr_lambda_fn_base(current_iteration) -> float:
        current_epoch = float(current_iteration) / iterations
        return 0.1 ** (current_epoch // opt.lr_update) / sqrt(nproc)

    def lr_lambda_fn_global(current_iteration) -> float:
        current_epoch = float(current_iteration) / iterations
        return 0.1 ** (current_epoch // opt.lr_update) / sqrt(nproc)

    logger.info(f'Initial LR set to: {initial_lr}.')
    module = model.module if isinstance(model, DistributedDataParallel) else model
    params_bert = module.txt_encoder.embed.parameters() if opt.text_type == 'bert' else []
    params_base = list(module.img_encoder.parameters()) + \
                  list(set(module.txt_encoder.parameters()) - set(params_bert))
    optimizer_base = optim.AdamW(params=params_base, lr=initial_lr)
    scheduler_base = lr_scheduler.LambdaLR(optimizer=optimizer_base, lr_lambda=lr_lambda_fn_base)
    if not use_global_optimizer(opt):
        return optimizer_base, scheduler_base, None, None, iterations
    else:
        params_global = list(set(module.parameters()) - set(params_bert) - set(params_base))
        optimizer_global = optim.Adamax(params=params_global, lr=initial_lr)
        scheduler_global = lr_scheduler.LambdaLR(optimizer=optimizer_global, lr_lambda=lr_lambda_fn_global)
        return optimizer_base, scheduler_base, optimizer_global, scheduler_global, iterations
