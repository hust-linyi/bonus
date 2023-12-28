import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain
        self.dataset = 'MO'    
        self.ratio = 0.05       # ratio of retained points annotation in each image
        self.description = ''

        self.raw_data_dir = '' # YOUR RAW DATA PATH
        self.processed_data_dir = '' # YOUR PROCESSED DATA PATH
        self.save_dir = '' # YOUR SAVE PATH
        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['pretrained'] = True
        self.model['out_c'] = 1

        # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = 0
        self.train['input_size'] = 224      # input size of the image
        self.train['train_epochs'] = 100     # number of training iterations
        self.train['batch_size'] = 64       # batch size
        self.train['checkpoint_freq'] = 999999  # epoch to save checkpoints
        self.train['lr'] = 0.0001           # initial learning rate
        self.train['weight_decay'] = 1e-4   # weight decay
        self.train['log_interval'] = 30     # iterations to print training results
        self.train['workers'] = 8        
        self.train['thresh'] = 0.65

        self.train['update_freq'] = 1
        self.train['max_growth_rate'] = 0.5
        self.train['bg_threshold'] = 0.1
        self.train['k'] = 3
        # --- data transform --- #
        self.transform = dict()

        # --- test parameters --- #
        self.test = dict()
        self.test['epoch'] = 'best'
        self.test['threshold'] = 0.65
        self.test['patch_size'] = 224
        self.test['overlap'] = 80

        # --- post processing --- #
        self.post = dict()
        self.post['max_area'] = 150
        self.post['min_area'] = 12

        self.radius = 15

    def parse(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--description', type=str, default='')
        parser.add_argument('--dataset', type=str, default='MO')
        parser.add_argument('--raw-data-dir', type=str, default=self.raw_data_dir)
        parser.add_argument('--processed-data-dir', type=str, default=self.processed_data_dir)
        parser.add_argument('--ratio', type=float, default=self.ratio, help='point ratio')
        parser.add_argument('--gpus', type=int, nargs='+', default=[0])        
        parser.add_argument('--update-freq', type=int, default=30)
        parser.add_argument('--epochs', type=int, default=self.train["train_epochs"])
        parser.add_argument('--k-neighbors', type=int, default=3)
        parser.add_argument('--threshold', type=float, default=self.test['threshold'], help='threshold to obtain the prediction from probability map')
        parser.add_argument('--random-seed', type=int, default=self.train['random_seed'], help='random seed for training')
        parser.add_argument('--lr', type=float, default=self.train['lr'])
        parser.add_argument('--batch-size', type=int, default=self.train['batch_size'])

        args = parser.parse_args()
        self.raw_data_dir = args.raw_data_dir
        self.processed_data_dir = args.processed_data_dir
        self.save_dir = args.save_dir
        self.description = args.description
        self.dataset = args.dataset
        self.ratio = args.ratio
        self.raw_data_dir = args.raw_data_dir
        self.processed_data_dir = args.processed_data_dir
        self.threshold = args.threshold
        
        self.random_seed = args.random_seed
        self.gpus = args.gpus

        self.train['update_freq'] = args.update_freq
        self.train["train_epochs"] = args.epochs
        self.train['k'] = args.k_neighbors
        self.train['lr'] = args.lr
        self.train['batch_size'] = args.batch_size
        self.train['data_dir'] = f'{args.processed_data_dir}/{args.dataset}/ratio{args.ratio}{args.description}/'
        self.train['thresh'] = args.threshold
        self.test['threshold'] = args.threshold

        self.r1 = 11
        self.r2 = 22
        self.gaussian_sigma = self.r1/4

        self.train["train_image_dir"] = f'{self.processed_data_dir}/{self.dataset}/images/train/'
        self.train["val_image_dir"] = f'{self.processed_data_dir}/{self.dataset}/images/val/'
        self.train["label_detect_dir"] = f'{self.processed_data_dir}/{self.dataset}/ratio{self.ratio}{self.description}/labels_detect/'
        self.train["label_bg_dir"] = f'{self.processed_data_dir}/{self.dataset}/ratio{self.ratio}{self.description}/labels_bg/'
        self.train['save_dir'] = f"{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/"
        self.test['image_dir'] = f'{self.processed_data_dir}/{self.dataset}/images/test/'
        self.test["label_dir"] = f'{self.raw_data_dir}/{self.dataset}/labels_point/'
        self.test['model_path'] = f'{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/checkpoints/checkpoint_{self.test["epoch"]}.pth.tar'
        self.test['save_dir'] = f"{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/{self.test['epoch']}"
            
        self.define_transforms()
        
    def define_transforms(self):
        self.transform['train'] = {
            # 'random_resize': [0.8, 1.25],
            'horizontal_flip': True,
            'vertical_flip': True,
            'random_crop': self.train['input_size'],
            'label_gaussian': (-1, self.gaussian_sigma),
            'to_tensor': 2,
            'normalize': np.load(f'{self.processed_data_dir}/{self.dataset}/images/mean_std.npy')
        }
        self.transform['val'] = {
            'to_tensor': 2,
            'normalize': np.load(f'{self.processed_data_dir}/{self.dataset}/images/mean_std.npy')
        }
        self.transform['test'] = {
            'to_tensor': 1,
            'normalize': np.load(f'{self.processed_data_dir}/{self.dataset}/images/mean_std.npy')
        }

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = f'{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/train_options.txt'
        else:
            filename = f'{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/{self.test["epoch"]}/test_options.txt'
        message = self._generate_message_from_options()
        with open(filename, 'w') as f:
            f.write(message)

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message