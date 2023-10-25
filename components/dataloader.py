import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from anatool import AnaLogger, AnaArgParser
from data import GloveImageCaptionDataset, Vocabulary
from data.dataset import BertImageCaptionDataset


def get_dataloader(opt, data_split, vocab, logger: AnaLogger):
    def collate_fn(data: list):
        """Build mini-batch tensors from a list of (image, caption) tuples.
        Args:
            data: list of (image, caption) tuple.
                - image: torch tensor of shape (36, 2048).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 36, 2048).
            padded_captions: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
            indices: list; sample index list.
        """
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, indices, img_ids = zip(*data)

        images = torch.stack(images, 0)

        lengths = [len(cap) for cap in captions]
        padded_captions = torch.full(
            (len(captions), max(lengths)),
            fill_value=0
        )
        for i, cap in enumerate(captions):
            padded_captions[i, :lengths[i]] = cap
        return images, padded_captions, torch.Tensor(list(lengths)).long(), torch.Tensor(list(indices)).long()

    if opt.text_type == 'glove':
        dataset = GloveImageCaptionDataset(
            opt=opt,
            data_split=data_split,
            vocab=vocab,
            logger=logger
        )
    else:
        dataset = BertImageCaptionDataset(
            opt=opt,
            data_split=data_split,
            logger=logger
        )
    if data_split == 'train' and opt.local_rank != -1:
        train_sampler = DistributedSampler(dataset=dataset)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=opt.train_batch_size,
            num_workers=opt.workers,
            pin_memory=opt.pin_memory,
            collate_fn=collate_fn,
            sampler=train_sampler
        )
        return train_loader, dataset

    elif data_split == 'train' and opt.local_rank == -1:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=opt.train_batch_size,
            num_workers=opt.workers,
            pin_memory=opt.pin_memory,
            collate_fn=collate_fn,
            shuffle=False
        )
        return train_loader, dataset

    elif data_split == 'dev':
        dev_loader = DataLoader(
            dataset=dataset,
            batch_size=opt.dev_batch_size,
            num_workers=opt.workers,
            pin_memory=opt.pin_memory,
            collate_fn=collate_fn,
            shuffle=False
        )
        return dev_loader, dataset

    elif data_split == 'test':
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=opt.test_batch_size,
            num_workers=opt.workers,
            pin_memory=opt.pin_memory,
            collate_fn=collate_fn,
            shuffle=False
        )
        return test_loader, dataset
    else:
        logger.error(f'Invalid data split: {data_split}.')
        raise ValueError


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    vocab = Vocabulary(opt=opt, logger=logger, load=True)
    dev_loader = get_dataloader(
        opt=opt,
        data_split='dev',
        vocab=vocab,
        logger=logger
    )
    for item in dev_loader:
        break
