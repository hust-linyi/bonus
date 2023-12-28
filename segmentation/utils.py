import os
import numpy as np
import torch
import time
import shutil

def split_forward_single(model, input, size, overlap):
    '''
    split the input image for forward passes
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), 1, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output

def split_forward_double(model, input, size, overlap):
    '''
    split the input image for forward passes
    '''
    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, _, h, w = input.size()

    output_seg = torch.zeros((b, 1, h, w)).cuda()
    output_edge = torch.zeros((b, 1, h, w)).cuda()

    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch_seg, output_patch_edge = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w

            patch_seg = output_patch_seg[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]
            patch_edge = output_patch_edge[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

            output_seg[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = patch_seg
            output_edge[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = patch_edge

    output_seg = output_seg[:,:,:h0,:w0].cuda()
    output_edge = output_edge[:,:,:h0,:w0].cuda()

    return output_seg, output_edge

# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def write_txt(results, filename, mode='w'):
    """ Save the result of losses and F1 scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num-1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num-1]))

def save_results(header, all_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for value in vals:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')

def save_checkpoint(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))

def save_bestcheckpoint(state, save_dir):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    torch.save(state, '{:s}/checkpoint_0.pth.tar'.format(cp_dir))

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        