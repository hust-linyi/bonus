"""
This script is used to prepare the dataset for training and testing.

"""

import os
import shutil
import numpy as np
from skimage import measure, morphology
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.ndimage import morphology as ndi_morph
from scipy.spatial import Voronoi
import json
import imageio
import utils
import math
from tqdm import tqdm
from shapely.geometry import Polygon

def main(opt, time=0):
    raw_image_dir = f'{opt.raw_data_dir}/{opt.dataset}/images'
    label_instance_dir = f'{opt.raw_data_dir}/{opt.dataset}/labels_instance/'
    label_complete_point_dir = f'{opt.raw_data_dir}/{opt.dataset}/labels_point/'
    label_partial_point_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_point/'
    label_detect_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_detect/'
    label_bg_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_bg/'
    label_vor_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_vor/'
    label_clu_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_clu/'
    label_patched_vor_dir = f'{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_vor/'
    label_patched_clu_dir = f'{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_clu/'

    processed_image_dir = f'{opt.processed_data_dir}/{opt.dataset}/images/'
    processed_detect_dir = f'{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_detect/'
    processed_bg_dir = f'{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_bg/'
    train_data_dir = f'{opt.processed_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/'
    probmap_dir = f"{opt.save_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/{opt.test['epoch']}/images_prob_maps/"
    pred_dir = f"{opt.save_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/{opt.test['epoch']}/images_pred/"

    data_list = utils.read_data_split(f'{opt.raw_data_dir}/{opt.dataset}/train_val_test.json')

    if time==0:
        create_point_label_from_instance(label_instance_dir, label_complete_point_dir)
        if opt.ratio <= 1:
            sample_points(label_complete_point_dir, f'{label_partial_point_dir}/{time}/', opt.ratio) 
            create_detect_label_from_points(f'{label_partial_point_dir}/{time}/', f'{label_detect_dir}/{time}/', data_list['train'], radius=opt.r1)
            split_patches(f'{label_detect_dir}/{time}/', processed_detect_dir, data_list['train'], post_fix='label_detect', patch_size=250, n=7)
        prepare_images(raw_image_dir, processed_image_dir, data_list) # cut training images into patches, move val/test images
        compute_mean_std(raw_image_dir, processed_image_dir, data_list['train']) 
    elif time>0:  
        curriculum_update(label_partial_point_dir, label_detect_dir, label_bg_dir, probmap_dir, time, opt, opt.test['label_dir'])
        split_patches(f'{label_bg_dir}/{time}', processed_bg_dir, data_list['train'], 'label_bg', patch_size=250, n=7)
        split_patches(f'{label_detect_dir}/{time}', processed_detect_dir, data_list['train'], 'label_detect', patch_size=250, n=7)
    else: # convert to vor/clu
        if opt.ratio <= 1:
            label_point_pred_dir = f'{opt.raw_data_dir}/{opt.dataset}/ratio{opt.ratio}{opt.description}/labels_point/pred/'
            create_point_label_from_detection(pred_dir, f'{label_partial_point_dir}/0/', label_point_pred_dir, data_list['train']+data_list['val'], opt.radius)
        else:
            label_point_pred_dir = label_complete_point_dir
        create_voronoi_label(label_point_pred_dir, label_vor_dir, data_list['train']+data_list['val'])
        create_cluster_label(raw_image_dir, label_point_pred_dir, label_vor_dir, label_clu_dir, data_list['train']+data_list['val'])
        split_patches(f'{processed_image_dir}/val', f'{processed_image_dir}/val_patch', data_list['val'], patch_size=250, n=7)
        split_patches(label_vor_dir, f'{label_patched_vor_dir}/train/', data_list['train'], post_fix="label_vor", patch_size=250, n=7)
        split_patches(label_vor_dir, f'{label_patched_vor_dir}/val_patch/', data_list['val'], post_fix="label_vor", patch_size=250, n=7)
        split_patches(label_clu_dir, f'{label_patched_clu_dir}/train/', data_list['train'], post_fix="label_clu", patch_size=250, n=7)
        split_patches(label_clu_dir, f'{label_patched_clu_dir}/val_patch/', data_list['val'], post_fix="label_clu", patch_size=250, n=7)

def create_point_label_from_instance(instance_dir, point_dir):
    utils.create_folder(point_dir)
    if len(os.listdir(point_dir)) > 0: # point label has been created
        return

    def get_point(img):
        a = np.where(img != 0)
        rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return (rmin + rmax) // 2, (cmin + cmax) // 2

    print("Generating point label from instance label...")
    for instance_name in os.listdir(instance_dir):
        name = instance_name.split('.')[0][:-6] # get rid of _label.npy

        instance_path = os.path.join(instance_dir, instance_name)
        instance = np.load(instance_path)
        h, w = instance.shape

        # extract bbox
        id_max = np.max(instance)
        label_point = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, id_max + 1):
            nucleus = instance == i
            if np.sum(nucleus) == 0:
                continue
            x, y = get_point(nucleus)
            label_point[x, y] = 255

        imageio.imsave(f'{point_dir}/{name}_label_point.png', label_point.astype(np.uint8))

def sample_points(label_complete_point_dir, label_partial_point_dir, ratio):
    print("sampling points")
    utils.create_folder(label_partial_point_dir)
    image_list = os.listdir(label_complete_point_dir)
    for imgname in image_list:
        if len(imgname) < 15 or imgname[-15:] != 'label_point.png':
            continue

        points = imageio.imread(f'{label_complete_point_dir}/{imgname}')
        if ratio < 1:
            points_labeled, N = measure.label(points, return_num=True)
            indices = np.random.choice(range(1, N + 1), int(N * ratio), replace=False)
            label_partial_point = np.isin(points_labeled, indices)
        else:
            label_partial_point = points

        imageio.imsave(f'{label_partial_point_dir}/{imgname}', (label_partial_point > 0).astype(np.uint8) * 255)

def create_detect_label_from_points(label_partial_point_dir, label_detect_dir, img_list, radius):
    utils.create_folder(label_detect_dir)

    for image_name in img_list:
        name = image_name.split('.')[0]
        points = imageio.imread(f'{label_partial_point_dir}/{name}_label_point.png')

        if np.sum(points > 0):
            label_detect = gaussian_filter(points.astype(float), sigma=radius/3)
            val = np.min(label_detect[points > 0])
            label_detect = label_detect / val
            label_detect[label_detect < 0.05] = 0
            label_detect[label_detect > 1] = 1
        else:
            label_detect = np.zeros(points.shape)
        imageio.imsave(f'{label_detect_dir}/{name}_label_detect.png', (label_detect*255).astype(np.uint8))

def create_point_label_from_detection(results_dir, label_original_point_dir, label_predicted_point_dir,
                                              train_list, dist_thresh):
    utils.create_folder(label_predicted_point_dir)
    print("Generating point label from detection results...")

    for image_name in train_list:
        name = image_name.split('.')[0]

        label_point = imageio.imread(f'{label_original_point_dir}/{name}_label_point.png') # Ground Truth partial label
        label_point_dilated = morphology.dilation(label_point, morphology.disk(dist_thresh))

        points_pred = imageio.imread(f'{results_dir}/{name}_pred.png') # detection result
        points_pred_labeled = measure.label(points_pred)
        point_regions = measure.regionprops(points_pred_labeled)

        ### add detected to partial label
        new_label_point = label_point.copy()
        for region in point_regions:
            x, y = int(region.centroid[0]), int(region.centroid[1])
            if label_point_dilated[x, y] > 0:
                continue
            else:
                new_label_point[x, y] = 255

        imageio.imsave(f'{label_predicted_point_dir}/{name}_label_point.png', new_label_point.astype(np.uint8))

def split_patches(data_dir, save_dir, data_list, post_fix=None, patch_size=250, n=7):
    """ split large image into small patches """
    utils.create_folder(save_dir)

    for image_name in data_list:
        image_path = f'{data_dir}/{image_name}_{post_fix}.png' if post_fix else f'{data_dir}/{image_name}.png'
        image = imageio.imread(image_path)

        assert(n!=1)
        seg_imgs = []
        h, w = image.shape[0], image.shape[1]
        h_overlap = math.ceil((n * patch_size - h) / (n-1))
        w_overlap = math.ceil((n * patch_size - w) / (n-1))
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                imageio.imsave(f'{save_dir}/{image_name}_{k}_{post_fix}.png', seg_imgs[k])
            else:
                imageio.imsave(f'{save_dir}/{image_name}_{k}.png', seg_imgs[k])

def prepare_images(raw_image_dir, processed_image_dir, data_list):
    utils.create_folder(f'{processed_image_dir}/train/')
    utils.create_folder(f'{processed_image_dir}/val/')
    utils.create_folder(f'{processed_image_dir}/test/')

    if len(os.listdir(f'{processed_image_dir}/train/')) == 0:
        split_patches(raw_image_dir, f'{processed_image_dir}/train/', data_list['train'])
    if len(os.listdir(f'{processed_image_dir}/val/')) == 0:
        for image_name in data_list['val']:
            shutil.copy(f'{raw_image_dir}/{image_name}.png', f'{processed_image_dir}/val/{image_name}.png')
    if len(os.listdir(f'{processed_image_dir}/test/')) == 0:
        for image_name in data_list['test']:
            shutil.copy(f'{raw_image_dir}/{image_name}.png', f'{processed_image_dir}/test/{image_name}.png')

def compute_mean_std(data_dir, train_data_dir, train_list):
    """ compute mean and standarad deviation of training images """
    if os.path.exists(f'{train_data_dir}/mean_std.npy'):
        return
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels
    print('Computing the mean and standard deviation of training data...')

    for file_name in train_list:
        img_name = f'{data_dir}/{file_name}.png'
        img = imageio.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3].astype(int)
        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel

    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    # normalization
    mean_values = mean_values / 255
    std_values = std_values / 255
    np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))
    np.savetxt('{:s}/mean_std.txt'.format(train_data_dir), np.array([mean_values, std_values]), '%.4f', '\t')
    print(f'mean std saved to {train_data_dir}' )

def curriculum_update(label_partial_point_dir, label_detect_dir, label_bg_dir, probmap_dir, time, opt, gt_dir=None):
    def get_centroid(mask):
        indicies = np.where(mask)
        rmin, rmax, cmin, cmax = indicies[0].min(), indicies[0].max(), indicies[1].min(), indicies[1].max()
        x,y = (rmin + rmax) // 2, (cmin + cmax) // 2
        return x,y

    utils.create_folder(f'{label_bg_dir}/{time}')
    utils.create_folder(f'{label_partial_point_dir}/{time}')
    utils.create_folder(f'{label_detect_dir}/{time}')
    
    # total_existing = 0
    # total_detected = 0
    # total_added = 0
    # total_num = 0
    # total_TP = 0
    # total_FP = 0
    # total_FN = 0

    for img_name in tqdm(os.listdir(f'{label_detect_dir}/{time-1}')):
        name = img_name.replace('_label_detect.png', '')
        if gt_dir is not None:
            gt = imageio.imread(f'{gt_dir}/{name}_label_point.png')

        probmap = imageio.imread(f'{probmap_dir}/{name}_prob.png').astype(float) / 255

        # update background label
        label_bg = (probmap < opt.train['bg_threshold']) * 255
        imageio.imsave(f'{label_bg_dir}/{time}/{name}_label_bg.png', label_bg.astype(np.uint8))

        points = imageio.imread(f'{label_partial_point_dir}/{time-1}/{name}_label_point.png') / 255
        detect = imageio.imread(f'{label_detect_dir}/{time-1}/{name}_label_detect.png') / 255

        binary_map = probmap > opt.train['thresh']
        labeled_areas = measure.label(binary_map)

        dist_scores = np.zeros((labeled_areas.max(),))
        conf_scores = np.zeros((labeled_areas.max(),))
        size_scores = np.zeros((labeled_areas.max(),))

        for i in range(1, labeled_areas.max()+1):
            area_mask = labeled_areas==i
            if area_mask.sum() > opt.post['max_area']:
                continue # too large
            if detect[area_mask].sum() != 0: continue # overlap with existing label
            if points.sum() != 0:
                existing_xs, existing_ys = np.where(points==1) # ([x0,x1,x2,...], [y0,y1,y2,...])
                x,y = get_centroid(area_mask)
                xs = np.zeros(existing_xs.shape) + x # [x,x,...,x,x]
                ys = np.zeros(existing_ys.shape) + y
                dist = np.sqrt((xs-existing_xs)*(xs-existing_xs) + (ys-existing_ys)*(ys-existing_ys)) # distance to all points
                dist_score = dist[np.argsort(dist)<opt.train['k']].mean() # mean distance to top k nearest neighbors
            else:
                dist_score = -1
            dist_scores[i-1] = dist_score
            conf_scores[i-1] = probmap[area_mask].mean()
            size_scores[i-1] = area_mask.sum()

        if points.sum() != 0:
            dist_scores = 1 - dist_scores/dist_scores.max()
        else:
            dist_scores = 1
        
        conf_scores = conf_scores/conf_scores.max()
        size_scores = 1 - size_scores/size_scores.max()       
        scores = dist_scores*conf_scores*size_scores

        sorted_idx = np.argsort(scores)
        sorted_idx = sorted_idx.max() - sorted_idx # reverse order

        prev_n_points = points.sum()
        n_valid_samples = len(scores.nonzero()[0])
        num = np.exp( -0.5 * prev_n_points/n_valid_samples ) * n_valid_samples # e ^ (- existed/detected)

        for i in range(int(num)):
            area_label = np.where(sorted_idx==i)[0][0] + 1
            area_mask = labeled_areas==area_label
            dilated_points = morphology.dilation(points, morphology.disk(opt.r1))
            if dilated_points[area_mask].sum() != 0: # overlap with candidates
                continue
            x,y = get_centroid(area_mask)
            points[x,y] = 1

        imageio.imsave(f'{label_partial_point_dir}/{time}/{name}_label_point.png', (points*255).astype(np.uint8))

        if np.sum(points > 0):
            label_detect = gaussian_filter(points.astype(float), sigma=opt.r1/3)
            val = np.min(label_detect[points > 0])
            label_detect = label_detect / val
            label_detect[label_detect < 0.05] = 0
            label_detect[label_detect > 1] = 1
        else:
            label_detect = np.zeros(points.shape)
        imageio.imsave(f'{label_detect_dir}/{time}/{name}_label_detect.png', (label_detect*255).astype(np.uint8))

        # if gt_dir is not None:
        #     TP, FP, FN = compute_accuracy(points, gt, radius=opt.r1, return_distance=False)
        #     total_TP += TP
        #     total_FP += FP
        #     total_FN += FN

        # total_existing += prev_n_points
        # total_detected += n_valid_samples
        # total_num += num
        # total_added += points.sum() - prev_n_points

        # print(f'"existing": {total_existing}, "detected": {total_detected}, "num": {total_num}, "added": {total_added}')
        # print(f"'TP': {total_TP}, 'FP': {total_FP}, 'FN': {total_FN}")

def create_voronoi_label(data_dir, save_dir, train_list):
    utils.create_folder(save_dir)
    if len(os.listdir(save_dir)) > 0:
        return
    print("Generating Voronoi label from point label...")

    for img_name in train_list:
        name = img_name.split('.')[0]

        img_path = '{:s}/{:s}_label_point.png'.format(data_dir, name)
        label_point = imageio.imread(img_path)
        h, w = label_point.shape

        points = np.argwhere(label_point > 0)
        # print(img_name, points.shape)
        if points.shape[0] < 4:
            print(f' Skipped Voronoi of {img_name} for too few points ({points.shape[0]})')
            edges = np.zeros((h, w), dtype=bool)
        else:
            vor = Voronoi(points)
            regions, vertices = utils.voronoi_finite_polygons_2d(vor)
            box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
            region_masks = np.zeros((h, w), dtype=np.int16)
            edges = np.zeros((h, w), dtype=bool)
            count = 1
            for region in regions:
                polygon = vertices[region]
                # Clipping polygon
                poly = Polygon(polygon)
                poly = poly.intersection(box)
                polygon = np.array([list(p) for p in poly.exterior.coords])
                if len(polygon.shape) == 1:
                    print(polygon.shape, {img_name}, 'too few indicies')
                    continue

                mask = utils.poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
                edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
                edges += edge
                region_masks[mask] = count
                count += 1

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(label_point, morphology.disk(2))
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        label_vor[:, :, 0] = morphology.closing(edges > 0, morphology.disk(1)).astype(np.uint8) * 255
        label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

        imageio.imsave('{:s}/{:s}_label_vor.png'.format(save_dir, name), label_vor)

def create_cluster_label(data_dir, label_point_dir, label_vor_dir, save_dir, train_list):
    utils.create_folder(save_dir)
    if len(os.listdir(save_dir)) > 0:
        return
    print("Generating cluster label from point label...")
    for img_name in train_list:
        name = img_name.split('.')[0]

        ori_image = imageio.imread('{:s}/{:s}.png'.format(data_dir, name))
        h, w, _ = ori_image.shape
        label_point = imageio.imread('{:s}/{:s}_label_point.png'.format(label_point_dir, name))

        # k-means clustering
        dist_embeddings = distance_transform_edt(255 - label_point).reshape(-1, 1)
        clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
        color_embeddings = np.array(ori_image, dtype=float).reshape(-1, 3) / 10
        embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

        # print("\t\tPerforming k-means clustering...")
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(embeddings)
        clusters = np.reshape(kmeans.labels_, (h, w))

        # get nuclei and background clusters
        overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
        nuclei_idx = np.argmax(overlap_nums)
        remain_indices = np.delete(np.arange(3), nuclei_idx)
        dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
        overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums)]

        nuclei_cluster = clusters == nuclei_idx
        background_cluster = clusters == background_idx

        # refine clustering results
        # print("\t\tRefining clustering results...")
        nuclei_labeled = measure.label(nuclei_cluster)
        initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
        refined_nuclei = np.zeros(initial_nuclei.shape, dtype=bool)

        label_vor = imageio.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name))
        voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
        voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

        # refine clustering results
        unique_vals = np.unique(voronoi_cells)
        cell_indices = unique_vals[unique_vals != 0]
        N = len(cell_indices)
        for i in range(N):
            cell_i = voronoi_cells == cell_indices[i]
            nucleus_i = cell_i * initial_nuclei

            nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
            nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
            nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
            refined_nuclei += nucleus_i_final > 0

        refined_label = np.zeros((h, w, 3), dtype=np.uint8)
        label_point_dilated = morphology.dilation(label_point, morphology.disk(10))
        refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(np.uint8) * 255
        refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

        imageio.imsave('{:s}/{:s}_label_clu.png'.format(save_dir, name), refined_label)


if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=True)
    main(opt)

