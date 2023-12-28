import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
from my_transforms import get_transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img


# get the image list pairs
def get_imgs_list(dir_list, post_fix=None):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label_vor.png', 'label_clu.png',...]
    :return: e.g. [(img1.png, img1_label_vor.png, img1_label_clu.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list

    img_filename_list = [os.listdir(_dir) for _dir in dir_list]
    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img), ]
        for i in range(1, len(img_filename_list)):
            img_name = f'{img1_name}_{post_fix[i - 1]}'
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


# dataset that supports multiple images
class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None, loader=img_loader):
        """
        :param dir_list: [img_dir, label_voronoi_dir, label_clu_dir]
        :param post_fix:  ['label_vor.png', 'label_clu.png']
        :param num_channels:  [3, 3, 3]
        :param data_transform: data transformations
        :param loader: image loader
        """
        super(DataFolder, self).__init__()

        if len(dir_list) != len(num_channels):
            raise (RuntimeError('Length of dir_list is different from length of num_channels.'))

        self.img_list = get_imgs_list(dir_list, post_fix)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

class CoarseDataset(data.Dataset):
    def __init__(self, img_dir, vor_dir, clu_dir, aug=False):
        img_list = []
        img_names = os.listdir(img_dir)
        for img_name in img_names:
            if not is_image_file(img_name):
                continue
            img = os.path.splitext(img_name)[0]
            img_list.append(img)
        self.img_list = img_list
        self.img_dir = img_dir
        self.vor_dir = vor_dir
        self.clu_dir = clu_dir
        self.aug = aug

        transform = {
            'train': {
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'vertical_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': 2,
                'to_tensor': 1
            },
            'val': {
                'random_crop': 224,
                'label_encoding': 2,
                'to_tensor': 1
            },
        }
        if aug:
            self.transform = get_transforms(transform['train'])
        else:
            self.transform = get_transforms(transform['val'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        img = Image.open(self.img_dir + '/' + img_name + '.png').convert('RGB')
        vor = Image.open(self.vor_dir + '/' + img_name + '_label_vor.png').convert('RGB')
        clu = Image.open(self.clu_dir + '/' + img_name + '_label_clu.png').convert('RGB')

        img, vor, clu = self.transform((img,vor,clu))
        return img, vor, clu

class GetInsAffinityLabelFromIndices():
    def __init__(self, indices_from, indices_to):
        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):
        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.greater(segm_label_from, -1), np.greater(segm_label_to, -1)) # filter out -1
        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)
        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class AffinityFromInstance(data.Dataset):
    def __init__(self, img_dir, aff_dir, vor_dir, clu_dir, indices_from, indices_to, aug=False):
        super().__init__()
        img_list = []
        img_name_list = os.listdir(img_dir)
        for img_name in img_name_list:
            if not is_image_file(img_name):
                continue
            img = os.path.splitext(img_name)[0] # remove .png
            img_list.append(img)

        self.img_list = img_list
        self.img_dir = img_dir
        self.aff_dir = aff_dir
        self.vor_dir = vor_dir
        self.clu_dir = clu_dir
        self.flip = aug

        self.extract_aff_lab_func = GetInsAffinityLabelFromIndices(indices_from, indices_to)

        transform = {
            'train': {
                'random_resize': [0.9, 1.25],
                'horizontal_flip': True,
                'vertical_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'label_encoding': 2,
                # 'to_tensor': 1
            },
            'val': {
                'random_crop': 224,
                'label_encoding': 2,
                # 'to_tensor': 1
            },
        }

        if aug:
            self.transform = get_transforms(transform['train'])
        else:
            self.transform = get_transforms(transform['val'])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(self.img_dir + '/' + img_name + '.png').convert('RGB')
        aff = np.load(self.aff_dir + '/' + img_name + '.npy')
        vor = Image.open(self.vor_dir + '/' + img_name + '_label_vor.png').convert('RGB')
        clu = Image.open(self.clu_dir + '/' + img_name + '_label_clu.png').convert('RGB')

        img = np.array(img)
        vor = np.array(vor)
        clu = np.array(clu)

        # label_encoding
        new_label = np.ones((vor.shape[0], vor.shape[1]), dtype=np.uint8) * 2
        new_label[vor[:, :, 0] > 255 * 0.3] = 0
        new_label[vor[:, :, 1] > 255 * 0.5] = 1
        new_label[(img[:, :, 0] == 0) * (img[:, :, 1] == 0) * (img[:, :, 2] == 0)] = 0
        vor = new_label.astype(np.uint8)

        new_label = np.ones((clu.shape[0], clu.shape[1]), dtype=np.uint8) * 2
        new_label[clu[:, :, 0] > 255 * 0.3] = 0
        new_label[clu[:, :, 1] > 255 * 0.5] = 1
        new_label[(img[:, :, 0] == 0) * (img[:, :, 1] == 0) * (img[:, :, 2] == 0)] = 0
        clu = new_label.astype(np.uint8)

        # random crop
        sh = np.random.randint(0, 250-224)
        sw = np.random.randint(0, 250-224)
        img = img[sh:sh+224, sw:sw+224, :]
        aff = aff[sh:sh+224, sw:sw+224]
        vor = vor[sh:sh+224, sw:sw+224]
        clu = clu[sh:sh+224, sw:sw+224]

        # flipping
        if self.flip:
            p = np.random.rand()
            if p < 0.5: # vertical flip
                img = img[::-1, :, :] 
                aff = aff[::-1, :]
                vor = vor[::-1, :]
                clu = clu[::-1, :]

            p = np.random.rand()
            if p < 0.5: # horizontal flip
                img = img[:, ::-1, :]
                aff = aff[:, ::-1]
                vor = vor[:, ::-1]
                clu = clu[:, ::-1]
        
        # to tensor
        img = torch.tensor(img.copy().transpose(2,0,1) / 255)
        bg_pos, fg_pos, neg = self.extract_aff_lab_func(aff.copy())
        vor = torch.tensor(vor.copy()).unsqueeze(0)
        clu = torch.tensor(clu.copy()).unsqueeze(0)

        return img, bg_pos, fg_pos, neg, vor, clu


class FullySupervisedDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, indices_from, indices_to, aug=False):
        super().__init__()
        img_list = []
        img_name_list = os.listdir(img_dir)
        for img_name in img_name_list:
            if not is_image_file(img_name):
                continue
            img = os.path.splitext(img_name)[0] # remove .png
            img_list.append(img)
        self.img_list = img_list

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.aug = aug
        self.extract_aff_lab_func = GetInsAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(self.img_dir + '/' + img_name + '.png').convert('RGB')
        label = np.load(self.label_dir + '/' + img_name + '.npy')

        img = np.array(img)

        # random crop
        sh = np.random.randint(0, 250-224)
        sw = np.random.randint(0, 250-224)
        img = img[sh:sh+224, sw:sw+224, :]
        label = label[sh:sh+224, sw:sw+224]

        # flipping
        if self.aug:
            p = np.random.rand()
            if p < 0.5: # vertical flip
                img = img[::-1, :, :] 
                label = label[::-1, :]

            p = np.random.rand()
            if p < 0.5: # horizontal flip
                img = img[:, ::-1, :]
                label = label[:, ::-1]

        # to tensor
        img = torch.tensor(img.copy().transpose(2,0,1).astype(np.float32) / 255)
        bg_pos, fg_pos, neg = self.extract_aff_lab_func(label.copy())
        seg = torch.tensor(label.copy() > 0).unsqueeze(0)

        return img, bg_pos, fg_pos, neg, seg # (152 1786)*3
