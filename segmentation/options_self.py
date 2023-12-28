import argparse
import utils

class Options():
    def __init__(self):
        self.fine_model_dir = '' # YOUR FINE MODEL PATH
        self.dataset = "MO"
        self.ratio = 2.00
        self.description = "_"
        self.data_dir = '' # YOUR DATA PATH
        self.label_dir = '' # YOUR LABEL PATH
        self.save_dir = '' # YOUR SAVE PATH
        self.gpus = [0]
        self.random_seed = 0

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=self.dataset)
        parser.add_argument('--ratio', type=float, default=self.ratio)
        parser.add_argument('--description', type=str, default=self.description)
        parser.add_argument('--data-dir', type=str, default=self.data_dir)
        parser.add_argument('--label-dir', type=str, default=self.label_dir)
        parser.add_argument('--save-dir', type=str, default=self.save_dir)
        parser.add_argument('--gpus', type=int, nargs='+', default=self.gpus)
        parser.add_argument('--random-seed', type=int, default=self.random_seed)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--data-thresh', type=float, default=0.55)
        parser.add_argument('--threshold', type=float, default=0.55)
        parser.add_argument('--aff-weight', type=float, default=0.1)
        parser.add_argument('--path-radius', type=int, default=8)
        args = parser.parse_args()

        self.dataset = args.dataset
        self.ratio = args.ratio
        self.description = args.description
        self.data_dir = args.data_dir
        self.label_dir = args.label_dir
        self.save_dir = args.save_dir
        self.fine_model_dir = f'{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/fine/checkpoints/checkpoint_0.pth.tar'
        self.gpus = args.gpus
        self.random_seed = args.random_seed

        self.data = {}
        self.data['thresh'] = args.data_thresh

        self.train = {}
        self.train['img_dir'] = f'{args.data_dir}/{args.dataset}/images/'
        self.train['aff_dir'] = f'{args.data_dir}/{args.dataset}/ratio{args.ratio}{args.description}/labels_aff_self/'
        self.train['save_dir'] = f'{args.save_dir}/{args.dataset}/ratio{args.ratio}{args.description}/self/'
        self.train['input_size'] = 224
        self.train['epochs'] = args.epochs
        self.train['lr'] = args.lr
        self.train['batch_size'] = args.batch_size
        self.train['scheduler_step'] = 30
        self.train['checkpoint_freq'] = 999999
        self.train['log_freq'] = 999999
        self.train['workers'] = 4
        self.train['weight'] = 0.5
        self.train['aff_weight'] = args.aff_weight
        self.train['path_radius'] = args.path_radius

        self.test = {}
        self.test['label_dir'] = args.label_dir
        self.test['thresh'] = args.threshold
        self.test['test_epoch'] = 0
        self.test['min_area'] = 20 
        self.test['overlap'] = 80

        utils.create_folder(self.train['save_dir'])

    def save_options(self):
        filename = f'{self.save_dir}/{self.dataset}/ratio{self.ratio}{self.description}/fine/options.txt'
        with open(filename, 'w') as file:
            file.write("# ---------- Options ---------- #\n")
            for group, options in self.__dict__.items():
                if type(options) == dict:
                    file.write('\n\n-------- {:s} --------\n'.format(group))
                    for k,v in options.items():
                        file.write(f'{k}: {v}\n')
                else:
                    file.write(f'{group}: {str(options)}\n')
