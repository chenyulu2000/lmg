import os

from anatool import AnaLogger, AnaArgParser


def get_data_path(opt, logger: AnaLogger):
    if opt.data_name == 'coco':
        data_dir = opt.coco_data_dir
    elif opt.data_name == 'f30k':
        data_dir = opt.f30k_data_dir
    else:
        logger.error(f'Invaliad dataset name: {opt.data_name}.')
        raise ValueError

    opt.vocab_word_idx_path = f'data/vocab_word_idx/{opt.text_type}_{opt.data_name}_word_idx.json'
    opt.glove_path = f'data/pretrained_weight/glove/glove_840B_{opt.data_name}_precomp.json.pkl'

    if opt.phase == 'train':
        opt.train_image_path = os.path.join(data_dir, opt.train_image_path)
        opt.train_cap_path = os.path.join(data_dir, opt.train_cap_path)
        opt.dev_image_path = os.path.join(data_dir, opt.dev_image_path)
        opt.dev_cap_path = os.path.join(data_dir, opt.dev_cap_path)

    if opt.phase == 'test':
        if opt.data_name == 'f30k' and opt.fold5:
            logger.error('Invalid fold5 value when testing f30k.')
            raise ValueError
        if opt.data_name == 'coco' and opt.fold5 != opt.test_coco_all:
            logger.error('Fold5 does not match test_coco_all.')
            raise ValueError
        if opt.data_name == 'coco' and opt.test_coco_all == True:
            opt.test_image_path = os.path.join(data_dir, opt.testall_image_path)
            opt.test_cap_path = os.path.join(data_dir, opt.testall_cap_path)
        else:
            opt.test_image_path = os.path.join(data_dir, opt.test_image_path)
            opt.test_cap_path = os.path.join(data_dir, opt.test_cap_path)
    return opt


if __name__ == '__main__':
    parser = AnaArgParser()
    opt = parser.cfg
    logger = AnaLogger(opt.exp_dir)
    opt = get_data_path(opt=opt, logger=logger)
    logger.info(opt)
