import os
import imageio 
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import skimage.morphology as morph
import scipy.ndimage as ndi
from PIL import Image
from tqdm import tqdm
from rich import print
from models import AffResUNet34
from utils import AverageMeter, save_results, split_forward_double, create_folder
from metrics import compute_metrics
from my_transforms import get_transforms
from skimage import measure
from options_fine import Options

def run(opt):
    print('========== evaluating fine model ==========')
    opt.save_options()

    img_dir = f"{opt.train['img_dir']}/test/"
    label_dir = opt.test['label_dir']
    save_dir = opt.train['save_dir']
    pred_dir = save_dir + '/pred/'
    create_folder(pred_dir)

    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = AverageMeter(len(metric_names))

    model = AffResUNet34()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    model_path = f'{opt.train["save_dir"]}/checkpoints/checkpoint_{opt.test["test_epoch"]}.pth.tar'
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    test_transform = get_transforms({'to_tensor': 1})

    img_names = os.listdir(img_dir)
    img_process = tqdm(img_names)
    for iname in img_process:
        img_name = iname[:-4] # remove .png
        img_process.set_description_str(f'=> Evaluating image {img_name}')

        img = Image.open(img_dir + '/' + img_name + '.png')
        img = test_transform((img,))[0].unsqueeze(0)

        seg, edge = split_forward_double(model, img, opt.train['input_size'], opt.test['overlap'])
        
        seg = torch.sigmoid(seg).squeeze()
        edge = torch.sigmoid(edge).squeeze()
        
        diff = seg-edge
        diff = diff.clamp(min=0, max=1).cpu().numpy()

        pred = np.zeros(diff.shape)
        pred[diff > opt.test['thresh']] = 1

        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, opt.test['min_area'])
        pred_labeled = ndi.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)
        pred_labeled = morph.dilation(pred_labeled)
        
        np.save(f'{pred_dir}/{img_name}', pred_labeled.astype(np.uint16))

        gt = np.load(f'{label_dir}/{img_name}_label.npy')
        metrics = compute_metrics(pred_labeled, gt, metric_names)

        test_results[img_name] = [metrics[mn] for mn in metric_names]

        all_result.update([metrics[mn] for mn in metric_names])

        print( 'Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn,res in zip(metric_names, all_result.avg)) )

    header = metric_names
    save_results(header, all_result.avg, test_results, f'{opt.train["save_dir"]}/test_results_fine.txt')

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
