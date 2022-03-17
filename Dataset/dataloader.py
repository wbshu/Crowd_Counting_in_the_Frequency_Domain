import numpy as np
from PIL import Image
import torch.utils.data as data
from glob import glob
import Dataset.dataprocessor as dp
from torchvision import transforms
import torch
import os
import math


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


class Collate_F():
    @staticmethod
    def train_collate(batch):
        batch = list(zip(*batch))
        images = torch.stack(batch[0], 0)
        chfs = torch.stack(batch[1], 0)
        return images, chfs


class CrowdData(data.Dataset):
    def __init__(self, img_path: str, dot_ann_path: str, mode: str, device: str = 'cuda', is_gray: bool = False,
                 min_size: int = 0, max_size: int = np.inf):
        '''

        Args:
            img_path ():
            dot_ann_path ():
            mode (): 'train', 'val', or 'test'
            device ():
            is_gray ():
            min_size ():  the image's shorter edge's min length
            max_size ():  the image's shorter edge's max length
        '''
        self.im_list = sorted(glob(os.path.join(img_path, '*.jpg')))
        self.dot_ann_list = sorted(glob(os.path.join(dot_ann_path, '*.npy')))
        self.mode = mode
        self.device = device
        self.is_gray = is_gray
        self.shorter_length_min = min_size
        self.shorter_length_max = max_size

        self.single_img_path = None
        self.single_dot_ann_path = None

        if self.is_gray:
            self.transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.people_counts = []
        self.dealt_imgs = []
        self.dealt_dotmap = []

        for item in range(0, len(self.im_list)):
            self.single_img_path = self.im_list[item]
            self.single_dot_ann_path = self.dot_ann_list[item]

            if self.is_gray:
                img = Image.open(self.single_img_path).convert('L')
            else:
                img = Image.open(self.single_img_path).convert('RGB')

            gt_data = np.load(self.single_dot_ann_path)
            if gt_data.shape[0] > 0:
                if gt_data.ndim == 1:
                    gt_data = np.expand_dims(gt_data[:2], axis=0)
                dot_ann = gt_data[:, :2]
            else:
                dot_ann = gt_data

            w, h = img.size
            if min([w, h]) < self.shorter_length_min:
                r = self.shorter_length_min / min([w, h])
                img, dot_ann = dp.Image_dotmap_processing.resize(img, dot_ann,
                                                                 np.ceil(np.array([w * r, h * r])).astype(int))
            if min([w, h]) > self.shorter_length_max:
                r = self.shorter_length_max / min([w, h])
                img, dot_ann = dp.Image_dotmap_processing.resize(img, dot_ann,
                                                                 np.ceil(np.array([w * r, h * r])).astype(int))

            self.process(img, dot_ann)

    def process(self, img, dot_ann):
        '''

        Args:
            img (PIL Image):  image
            gt_data (ndarray):  dotted_annotation with or without other information

        Returns: deal with image and dotted_annotation, and put the dealt result into dealt_imgs/dealt_dotmap/people_count

        '''
        raise NotImplemented

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        if self.mode.lower().startswith('train'):
            return self.dealt_imgs[item], self.dealt_dotmap[item]
        else:
            return self.dealt_imgs[item], self.dealt_dotmap[item], self.people_counts[item], self.im_list[item]


class CrowdData_Harddish_Load(data.Dataset):
    def __init__(self, img_path: str, dot_ann_path: str, mode: str, device: str = 'cuda', is_gray: bool = False,
                 min_size: int = 0, max_size: int = np.inf):
        '''

        Args:
            img_path ():
            dot_ann_path ():
            mode (): 'train', 'val', or 'test'
            device ():
            is_gray ():
            min_size ():  the image's shorter path's min length
            max_size ():  the image's shorter path's max length
        '''
        self.im_list = sorted(glob(os.path.join(img_path, '*.jpg')))
        self.dot_ann_list = sorted(glob(os.path.join(dot_ann_path, '*.npy')))
        self.mode = mode
        self.device = device
        self.is_gray = is_gray
        self.shorter_length_min = min_size
        self.shorter_length_max = max_size

        if self.is_gray:
            self.transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.people_counts = []
        self.dealt_imgs = []
        self.dealt_dotmap = []

        for item in range(0, len(self.im_list)):
            img = self.im_list[item]
            dot_ann = self.dot_ann_list[item]

            self.process(img, dot_ann)

    def process(self, img, dot_ann):
        '''

        Args:
            img (PIL Image):  image
            gt_data (ndarray):  dotted_annotation with or without other information

        Returns: deal with image and dotted_annotation, and put the dealt result into dealt_imgs/dealt_dotmap/people_count

        '''
        raise NotImplemented

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        if self.mode.lower().startswith('train'):
            return self.dealt_imgs[item], self.dealt_dotmap[item]
        else:
            return self.dealt_imgs[item], self.dealt_dotmap[item], self.people_counts[item], self.im_list[item]


class ChfData_RCrop(CrowdData):
    def __init__(self, img_path: str, dot_ann_path: str, mode: str, img_size: int = 384, chf_step: int = 30,
                 chf_tik: float = 0.01, min_size: int = 0, max_size: int = np.inf, bandwidth: int = 8,
                 device: str = 'cuda', is_gray: bool = False):
        '''

        Args:
            img_path ():
            dot_ann_path ():
            mode ():
            img_size (int): the input image size of the DNN.
            chf_step (int): sample number of ch.f. in each direction (four direction: x+,x-,y+,y-)
            chf_tik (float): sample interval between two sample points of ch.f.
            max_size (int): the image's shorter edge's max length, if exceed, it will be contracted proportionally
            bandwidth ():
            device ():
            is_gray ():
        '''

        assert img_size <= max_size
        self.input_img_size = img_size
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.bandwidth = bandwidth

        if mode.lower().startswith('train'):
            super(ChfData_RCrop, self).__init__(img_path, dot_ann_path, mode, device, is_gray,
                                                max([self.input_img_size, min_size]),
                                                max_size)
        else:
            super(ChfData_RCrop, self).__init__(img_path, dot_ann_path, mode, device, is_gray, 0,
                                                max_size)

    def process(self, img, dot_ann):
        if self.mode.lower().startswith('train'):
            dot_ann = torch.from_numpy(dot_ann).to(device=self.device)
        else:
            self.people_counts.append(dot_ann.shape[0])
        img = self.transform(img)
        self.dealt_imgs.append(img)
        self.dealt_dotmap.append(dot_ann)

    def __getitem__(self, item):
        if self.mode.lower().startswith('train'):
            # random crop and rondom flip
            img, dot_ann = dp.ImgTensor_dotTensor_processing.random_crop(self.dealt_imgs[item].to(device=self.device),
                                                                         self.dealt_dotmap[item],
                                                                         self.input_img_size,
                                                                         dp.ImgTensor_dotTensor_processing.crop)
            img, dot_ann = dp.ImgTensor_dotTensor_processing.random_mirrow(img, dot_ann)

            return img, dot_ann.to(device=self.device)
        else:
            return self.dealt_imgs[item], self.dealt_dotmap[item], self.people_counts[item], self.im_list[item]


class ChfData_RCrop_Harddish_Load(CrowdData_Harddish_Load):
    def __init__(self, img_path: str, dot_ann_path: str, mode: str, img_size: int = 384, chf_step: int = 30,
                 chf_tik: float = 0.01, min_size: int = 0, max_size: int = np.inf, bandwidth: int = 8,
                 device: str = 'cuda', is_gray: bool = False):
        '''

        Args:
            img_path ():
            dot_ann_path ():
            mode ():
            img_size (int): the input image size of the DNN.
            chf_step (int): sample number of ch.f. in each direction (four direction: x+,x-,y+,y-)
            chf_tik (float): sample interval between two sample points of ch.f.
            max_size (int): the image's shorter path's max length, if exceed, it will be contracted proportionally
            bandwidth ():
            device ():
            is_gray ():
        '''

        assert img_size <= max_size
        self.input_img_size = img_size
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.bandwidth = bandwidth
        if mode.lower().startswith('train'):
            super(ChfData_RCrop_Harddish_Load, self).__init__(img_path, dot_ann_path, mode, device, is_gray,
                                                              max([self.input_img_size, min_size]), max_size)
        else:
            super(ChfData_RCrop_Harddish_Load, self).__init__(img_path, dot_ann_path, mode, device, is_gray, 0,
                                                              max_size)

    def process(self, img, dot_ann):
        pass

    def __getitem__(self, item):
        # load img/dotmap from hard dish
        single_img_path = self.im_list[item]
        single_dot_ann_path = self.dot_ann_list[item]

        if self.is_gray:
            img = Image.open(single_img_path).convert('L')
        else:
            img = Image.open(single_img_path).convert('RGB')

        gt_data = np.load(single_dot_ann_path)
        if gt_data.shape[0] > 0:
            if gt_data.ndim == 1:
                gt_data = np.expand_dims(gt_data[:2], axis=0)
            dot_ann = gt_data[:, :2]
        else:
            dot_ann = gt_data

        if self.mode.lower().startswith('train'):    ### for train
            # random crop and flip
            img, dot_ann = dp.Image_dotmap_processing.random_crop(img, dot_ann, self.input_img_size)
            img, dot_ann = dp.Image_dotmap_processing.random_mirrow(img, dot_ann)
            img = self.transform(img)
            dot_ann = torch.from_numpy(dot_ann)

            return img, dot_ann
        else:             ### for val
            img = self.transform(img)
            dot_ann = torch.from_numpy(dot_ann)

            return img, dot_ann, dot_ann.shape[0], self.im_list[item]


class NWPU_Test_Loader(data.Dataset):
    def __init__(self, img_path: str, min_size: int = 0, max_size: int = np.inf):
        '''

        Args:
            img_path ():
            min_size ():  the image's shorter edge's min length
            max_size ():  the image's shorter edge's max length
        '''
        self.im_list = sorted(glob(os.path.join(img_path, '*.jpg')))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.shorter_length_min = min_size
        self.shorter_length_max = max_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.name = []

        for item in range(0, len(self.im_list)):
            single_img_path = self.im_list[item]
            self.name.append(os.path.split(single_img_path)[1].split('.')[0])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        single_img_path = self.im_list[item]
        img = Image.open(single_img_path).convert('RGB')

        w, h = img.size
        if min([w, h]) < self.shorter_length_min:
            r = self.shorter_length_min / min([w, h])
            img = img.resize((int(math.ceil(w * r)), int(math.ceil(h * r))))
        elif min([w, h]) > self.shorter_length_max:
            r = self.shorter_length_max / min([w, h])
            img = img.resize((int(math.ceil(w * r)), int(math.ceil(h * r))))

        img = self.transform(img)

        return img, self.name[item]
