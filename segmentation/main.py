import os
import torch
import numpy as np
import random
from options import Options

from train import main as train
from test import main as test

def main():
    opt = Options(isTrain=True)
    opt.parse()

    torch.set_num_threads(1)
    if opt.train['random_seed'] >= 0:
        print('=> Using random seed {:d}'.format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpus)

    opt.print_options()

    print('=> Preparing training samples')
    prepare_data(opt)

    if opt.ratio <= 1:
        print("training")
        train(opt)

        # evaluation on test data
        opt.test['image_dir'] = f'{opt.processed_data_dir}/{opt.dataset}/images/test/'
        test(opt, eval_flag=True)

        # inference on all data
        opt.test['image_dir'] = f'{opt.raw_data_dir}/{opt.dataset}/images/'
        test(opt, eval_flag=False)

    prepare_data(opt, time=-1) # convert to vor/clu

if __name__ == '__main__':
    main()
