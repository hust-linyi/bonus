"""
This script is used to test the trained model using test dataset and produce final
segmentation maps.

Params:
    test_dir: the path to the folder of test images
    label_dir: the path to the folder of ground-truth labels
    save_dir: the path to save results
    model_path: the path to the trained mode
    mean_file_path: the path to the mean and std file

Outputs:
    The image results will be saved in two folders under model_path:
    -test_prob_maps: the probability maps
    -test_segmentation: the segmentation results
    The values of all metrics will be saved as .txt files under save_dir:
    -test_avg_result.txt: the average values of evaluation metrics:  F1 score, Dice, AJI
    -test_all_results.txt: the values of metrics for each image

Author: Hui Qu
"""

import os
import torch
import numpy as np
from PIL import Image
import skimage.morphology as ski_morph
from skimage import measure
import imageio
from model import create_model
import utils
from my_transforms import get_transforms

def main(opt, eval_flag=False):
    opt.isTrain = False

    img_dir = opt.test['image_dir']
    label_dir = opt.test['label_dir']
    model_path = opt.test['model_path']
    save_dir = opt.test['save_dir']
    
    utils.create_folder(save_dir)
    opt.save_options()
    
    if eval_flag:
        test_results = dict()
        total_TP = 0.0
        total_FP = 0.0
        total_FN = 0.0
        total_d_list = []

    test_transform = get_transforms(opt.transform['test'])

    model_name = opt.model['name']
    model = create_model(model_name, opt.model['out_c'], opt.model['pretrained'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    model = model.module
    model.eval()

    subset = img_dir.split('/') 
    subset = subset[-1] if len(subset[-1])>0 else subset[-2] # PATH_TO_SOMEWHERE/subset/ -> "subset"
    prob_maps_folder = f'{save_dir}/{subset}_prob_maps'
    pred_folder = f'{save_dir}/{subset}_pred'
    utils.create_folder(prob_maps_folder)
    utils.create_folder(pred_folder)

    for img_name in os.listdir(img_dir):
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        if eval_flag:
            label_path = f'{label_dir}/{name}_label_point.png'
            gt = imageio.imread(label_path)

        input = test_transform((img,))[0].unsqueeze(0)

        prob_maps = get_probmaps(input, model, opt)

        pred = prob_maps > opt.test['threshold']
        pred_labeled, N = measure.label(pred, return_num=True)
        if N > 1:
            bg_area = ski_morph.remove_small_objects(pred_labeled, opt.post['max_area']) > 0
            large_area = ski_morph.remove_small_objects(pred_labeled, opt.post['min_area']) > 0
            pred = pred * (bg_area==0) * (large_area>0)

        if eval_flag:
            TP, FP, FN, d_list = utils.compute_accuracy(pred, gt, radius=opt.r1, return_distance=True)
            # print(TP, FP, FN)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_d_list += d_list

            test_results[name] = [float(TP)/(TP+FN+1e-8), float(TP)/(TP+FP+1e-8), float(2*TP)/(2*TP+FP+FN+1e-8)]

        imageio.imsave(f'{pred_folder}/{name}_pred.png', pred.astype(np.uint8) * 255)
        imageio.imsave(f'{prob_maps_folder}/{name}_prob.png', (prob_maps*255).astype(np.uint8))

    if eval_flag:
        recall = float(total_TP) / (total_TP + total_FN + 1e-8)
        precision = float(total_TP) / (total_TP + total_FP + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        if len(total_d_list) > 0:
            mu = np.mean(np.array(total_d_list))
            sigma = np.sqrt(np.var(np.array(total_d_list)))
        else:
            mu = -1
            sigma = -1

        print('Average: precision\trecall\tF1\tmean\tstd:'
              '\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:3f}\t{:.3f}'.format(precision, recall, F1, mu, sigma))

        header = ['precision', 'recall', 'F1', 'mean', 'std']
        save_results(header, [precision, recall, F1, mu, sigma], test_results, f'{save_dir}/{subset}_test_result_{opt.test["threshold"]}.txt')

def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])
    output = torch.sigmoid(output[0,0,:,:]).cpu().numpy()

    return output


def save_results(header, avg_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_result)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_result[i]))
        file.write('{:.4f}\n'.format(avg_result[N - 1]))

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\t'.format(key))
            for i in range(len(vals)-1):
                file.write('{:.4f}\t'.format(vals[i]))
            file.write('{:.4f}\n'.format(vals[-1]))


if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=False)
    opt.parse()
    main(opt)