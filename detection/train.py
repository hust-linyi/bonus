import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
from PIL import Image
import numpy as np
from skimage import measure
import skimage.morphology as ski_morph
import imageio as io
from model import create_model
import utils
from dataset import DataFolder
from my_transforms import get_transforms
from test import main as test
from prepare_data import main as update_data

def main(opt):
    bg_flag = False
    best_score = -1
    update_time = 0
    opt.isTrain = True

    utils.create_folder(opt.train['save_dir'])

    opt.define_transforms()
    opt.save_options()

    model_name = opt.model['name']
    model = create_model(model_name, opt.model['out_c'], opt.model['pretrained'])
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    criterion = torch.nn.MSELoss(reduction='none').cuda()


    dir_list = [opt.train["train_image_dir"], opt.train["label_detect_dir"]]
    post_fix = ['label_detect.png']
    num_channels = [3, 1]
    train_transform = get_transforms(opt.transform['train'])

    train_set = DataFolder(dir_list, post_fix, num_channels, train_transform, r=opt.r2)
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])
    val_transform = get_transforms(opt.transform['val'])

    num_epoch = opt.train['train_epochs']

    for epoch in range(num_epoch):
        print(f'Epoch: [{epoch+1}/{num_epoch}]')
        train_loss = train(opt, train_loader, model, optimizer, criterion, with_bg=bg_flag)

        # evaluate on val set
        with torch.no_grad():
            val_recall, val_prec, val_F1 = validate(opt, model, val_transform)

        # check if it is the best accuracy
        is_best = val_F1 > best_score
        best_score = max(val_F1, best_score)
        print(f'Best Score: {best_score}')
        cp_flag = True if (epoch + 1) % opt.train['checkpoint_freq'] == 0 else False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, epoch, is_best, opt.train['save_dir'], cp_flag)

        if (epoch+1) % opt.train['update_freq'] == 0:
            print('updating label')
            update_time += 1

            # run inference on all data
            opt.test['image_dir'] = f'{opt.raw_data_dir}/{opt.dataset}/images' # change to all images
            test(opt)

            update_data(opt, time=update_time)

            if (epoch+1) == opt.train['update_freq']: # load background label
                bg_flag = True
                dir_list = [opt.train["train_image_dir"], opt.train["label_detect_dir"], opt.train['label_bg_dir']]
                post_fix = ['label_detect.png', 'label_bg.png']
                num_channels = [3, 1, 1]
                train_set = DataFolder(dir_list, post_fix, num_channels, train_transform, r=opt.r2)
                train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                                        num_workers=opt.train['workers'])

def train(opt, train_loader, model, optimizer, criterion, with_bg=False):
    results = utils.AverageMeter(1)

    model.train()

    for i, sample in enumerate(train_loader):
        img = sample[0].cuda()
        fg_label = sample[1].squeeze(1).cuda()
        mask = sample[-1].squeeze(1).cuda()
        if with_bg:
            bg = sample[2].squeeze(1).cuda()

        output = model(img).squeeze(1)
        probmaps = torch.sigmoid(output)

        if with_bg:
            mask = (mask + bg) > 0

        weight_map = mask.float().clone()
        weight_map[fg_label > 0] = 10

        loss_map = criterion(probmaps, fg_label)
        loss = torch.sum(loss_map * weight_map) / mask.sum()

        result = [loss.item(),]
        results.update(result, img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del img, output, loss

        if i % opt.train['log_interval'] == 0:
            print('\tIteration: [{:d}/{:d}]\tLoss {r[0]:.4f}'.format(i, len(train_loader), r=results.avg))

    print('\t=> Train Avg: Loss {r[0]:.4f}'.format(r=results.avg))

    return results.avg[0]


def validate(opt, model, data_transform):
    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0

    model.eval()

    img_dir = opt.train['val_image_dir']
    label_dir = opt.test['label_dir']

    img_names = os.listdir(img_dir)
    for img_name in img_names:
        # load test image
        img_path = f'{img_dir}/{img_name}'
        img = Image.open(img_path)
        name = img_name.split('.')[0]

        label_path = '{:s}/{:s}_label_point.png'.format(label_dir, name)
        gt = io.imread(label_path)

        img, label = data_transform((img, Image.fromarray(gt)))
        img = img.unsqueeze(0)

        prob_map = get_probmaps(img, model, opt)
        prob_map = prob_map.cpu().numpy()
        pred = prob_map > opt.test['threshold']  # prediction
        pred_labeled, N = measure.label(pred, return_num=True)
        if N > 1:
            bg_area = ski_morph.remove_small_objects(pred_labeled, opt.post['max_area']) > 0
            large_area = ski_morph.remove_small_objects(pred_labeled, opt.post['min_area']) > 0
            pred = pred * (bg_area==0) * (large_area>0)

        TP, FP, FN = utils.compute_accuracy(pred, gt, radius=opt.r1)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    recall = float(total_TP) / (total_TP + total_FN + 1e-8)
    precision = float(total_TP) / (total_TP + total_FP + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    print('\t=> Val Avg:\tRecall {:.4f}\tPrec {:.4f}\tF1 {:.4f}'.format(recall, precision, F1))

    return recall, precision, F1


def get_probmaps(img, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(img.cuda())
    else:
        output = utils.split_forward(model, img, size, overlap)
    output = output.squeeze(0)
    prob_maps = torch.sigmoid(output[0,:,:])

    return prob_maps


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


if __name__ == '__main__':
    main()
