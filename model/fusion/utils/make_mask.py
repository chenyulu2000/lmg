import torch

from anatool import AnaArgParser, AnaLogger


# making the sequence mask
def make_mask(feature):
    """
    :param feature:
        for img: (bs, proposals, embedding_size)
        for text - do text.unsqueeze(2) first : (bs, seq_len, 1)
    :return:
        shape: (bs, 1, 1, seq_len/proposal)
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


if __name__ == '__main__':
    from components import get_data_path

    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
    feature = torch.rand(32, 36, 1024)
    logger.debug(make_mask(feature).size())  # (32, 1, 1, 36)
