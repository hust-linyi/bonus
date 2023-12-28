import os
import imageio
import torch
import shutil
import random
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
from skimage import measure
from torch.utils.data import DataLoader
from rich import print
from dataset import FullySupervisedDataset
from models import AffResUNet34, AffinityLoss
from indexing import PathIndex
from utils import AverageMeter, split_forward_double, save_checkpoint, save_bestcheckpoint, create_folder
from my_transforms import get_transforms
from options_self import Options

def run(opt):
    prepare_data(opt)
    print(' ========== Training Self Stage ==========')
    opt.save_options()

    create_folder(opt.train['save_dir'])

    crop_size = opt.train['input_size'] # 224
    path_index = PathIndex(radius=opt.train['path_radius'], default_size=(crop_size,crop_size))
    
    model = AffinityLoss(path_index)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    train_set = FullySupervisedDataset(
        img_dir = opt.train['img_dir'] + '/train/',
        label_dir = opt.train['aff_dir'] + '/train/',
        indices_from = path_index.src_indices,
        indices_to = path_index.dst_indices,
        aug = True,
    )

    val_set = FullySupervisedDataset(
        img_dir = opt.train['img_dir'] + '/val_patch/',
        label_dir = opt.train['aff_dir'] + '/val_patch/',
        indices_from = path_index.src_indices,
        indices_to = path_index.dst_indices,
        aug = False
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=opt.train['batch_size'], num_workers=opt.train['workers'])
    val_loader = DataLoader(val_set, shuffle=False, batch_size=opt.train['batch_size'], num_workers=opt.train['workers'])
    
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99), weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.train['scheduler_step'], gamma=0.1)
    criterion = nn.BCELoss().cuda()

    num_epoch = opt.train['epochs']
    print("=> Initial learning rate: {:g}".format(opt.train['lr']))
    print("=> Batch size: {:d}".format(opt.train['batch_size']))
    print("=> Training epochs: {:d}".format(opt.train['epochs']))

    min_loss = 100
    for ep in range(num_epoch):
        print('Epoch: [{:d}/{:d}]'.format(ep+1, num_epoch))

        train_results = train(train_loader, model, optimizer, criterion, opt)
        train_loss, loss_seg, loss_aff = train_results

        state = {'epoch': ep+1, 'state_dict': model.state_dict()}

        cp_flag = (ep + 1) % opt.train['checkpoint_freq'] == 0
        save_checkpoint(state, ep, opt.train['save_dir'], cp_flag)

        scheduler.step()

        val_loss = val(val_loader, model, criterion, opt)
        print('val_loss:', val_loss, 'min_loss:', min_loss)
        if val_loss < min_loss:
            min_loss = val_loss
            save_bestcheckpoint(state, opt.train['save_dir'])

def prepare_data(opt):
    print('========== preparing data ==========')
    create_folder(f"{opt.train['aff_dir']}/train/")
    create_folder(f"{opt.train['aff_dir']}/val_patch/")
    if len(os.listdir(f"{opt.train['aff_dir']}/train/")) > 0 and len(os.listdir(f"{opt.train['aff_dir']}/val_patch/")) > 0:
        return
        
    model = AffResUNet34()
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    model_path = opt.fine_model_dir
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"=> loaded model at epoch {checkpoint['epoch']}")

    for subset in ['train','val_patch']:
        aff_dir = f"{opt.train['aff_dir']}/{subset}/"
    
        img_dir = opt.train['img_dir'] + '/' + subset + '/'
        for img_name in os.listdir(img_dir):
            img = imageio.imread(f'{img_dir}/{img_name}')
            img = torch.tensor(np.array(img).transpose(2,0,1)/255).unsqueeze(0).float()

            with torch.no_grad():
                seg,edge = split_forward_double(model, img, 224, 80)
            seg = torch.sigmoid(seg).squeeze()
            edge = torch.sigmoid(edge).squeeze()

            diff = seg - edge
            diff = diff.clamp(min=0, max=1).cpu().numpy()

            pred = diff > opt.data['thresh']

            pred_labeled = measure.label(pred)
            pred_labeled = morph.remove_small_objects(pred_labeled, opt.test['min_area'])
            pred_labeled = ndi.binary_fill_holes(pred_labeled > 0)
            pred_labeled = measure.label(pred_labeled)
            pred_labeled = morph.dilation(pred_labeled)
            
            np.save(f'{aff_dir}/{img_name[:-4]}.npy', pred_labeled)

def train(train_loader, model, optimizer, criterion, opt):
    results = AverageMeter(3)
    model.train()
    for i, sample in enumerate(train_loader):
        img = sample[0].float().cuda()
        bg_pos_label = sample[1].float().cuda()
        fg_pos_label = sample[2].float().cuda()
        neg_label = sample[3].float().cuda()
        seg_label = sample[4].float().cuda()

        seg, edge = model(img)
        seg = torch.sigmoid(seg)
        edge = torch.sigmoid(edge)

        pos_aff_loss, neg_aff_loss = model.module.to_loss(edge)
        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)
        loss_aff = pos_aff_loss + neg_aff_loss
        loss_aff *= opt.train['aff_weight']

        loss_seg = criterion(seg, seg_label)
        total_loss = loss_aff + loss_seg

        results.update([total_loss.item(), loss_seg.item(), loss_aff.item()], img.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % opt.train['log_freq'] == 0:
            print('Iteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_seg {r[1]:.4f}'
                        '\tLoss_aff {r[2]:.4f}'.format(i, len(train_loader), r=results.avg))
    print('\t=> Train Avg: Loss {r[0]:.4f}'
            '\tloss_seg {r[1]:.4f}'
            '\tloss_aff {r[2]:.4f}'.format(r=results.avg))

    return results.avg

def val(val_loader, model, criterion, opt):
    model.eval()
    results = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            img = sample[0].float().cuda()
            bg_pos_label = sample[1].float().cuda()
            fg_pos_label = sample[2].float().cuda()
            neg_label = sample[3].float().cuda()
            seg_label = sample[4].float().cuda()

            seg, edge = model(img)
            seg = torch.sigmoid(seg)
            edge = torch.sigmoid(edge)

            pos_aff_loss, neg_aff_loss = model.module.to_loss(edge)
            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)
            loss_aff = pos_aff_loss + neg_aff_loss
            loss_aff *= opt.train['aff_weight']

            loss_seg = criterion(seg, seg_label)

            result = loss_seg.item() + loss_aff.item()
            results += result

        val_loss = results / len(val_loader)
    return val_loss

if __name__ == '__main__':
    opt = Options()
    opt.parse()
    if opt.random_seed >= 0:
        print('=> Using random seed {:d}'.format(opt.random_seed))
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed(opt.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.random_seed)
        random.seed(opt.random_seed)
    else:
        torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpus)
    run(opt)
