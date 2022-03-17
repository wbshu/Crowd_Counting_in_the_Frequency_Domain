from scipy.io import loadmat
import glob
import numpy as np
import os
from PIL import Image
from Dataset.dataprocessor import Image_dotmap_processing as idp
from Dataset.dataloader import cal_new_size


class Trans_gt_to_ndarray:
    @staticmethod
    def trans_ann_to_npy_SHTC(target_path: str, save_path: str):
        """

        Args:
            target_path (str): the directory where the original annotation maps are
            save_path (str): the directory where the npy annotation maps will be put

        Returns:

        """
        assert target_path != save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in glob.glob(target_path + '/*.mat'):
            dot = loadmat(file)
            x = dot[list(dot.keys())[-1]][0, 0]['location'][0, 0].astype(np.float)
            np.save(''.join((save_path, '/', os.path.split(file)[-1].split('.')[0], '.npy')), x)

    @staticmethod
    def trans_ann_to_npy_QNRF(target_path: str, save_path: str):
        """

        Args:
            target_path (str): the directory where the original annotation maps are
            save_path (str): the directory where the npy annotation maps will be put

        Returns:

        """
        assert target_path != save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in glob.glob(target_path + '/*.mat'):
            dot = loadmat(file)
            x = dot[list(dot.keys())[-1]].astype(np.float32)
            np.save(''.join((save_path, '/', os.path.split(file)[-1].split('.')[0], '.npy')), x)

    @staticmethod
    def trans_ann_to_npy_JHU(target_path: str, save_path: str):
        '''

        Args:
            target_path (str): the directory where the original annotation maps are
            save_path (str): the directory where the npy annotation maps will be put

        Returns:

        '''
        assert target_path != save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in glob.glob(target_path + '/*.txt'):
            x = np.loadtxt(file).astype(np.float32)
            np.save(''.join((save_path, '/', os.path.split(file)[-1].split('.')[0], '.npy')), x)

    @staticmethod
    def trans_ann_to_npy_NWPU(target_path: str, save_path: str):
        """

        Args:
            target_path (str): the directory where the original annotation maps are
            save_path (str): the directory where the npy annotation maps will be put

        Returns:

        """
        assert target_path != save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in glob.glob(target_path + '/*.mat'):
            dot = loadmat(file)
            x = dot[list(dot.keys())[-2]].astype(np.float32)
            np.save(''.join((save_path, '/', os.path.split(file)[-1].split('.')[0], '.npy')), x)


class Directory_path:
    @staticmethod
    def prefix_suffix(dataset_name: str):
        if dataset_name.lower().startswith('jhu'):
            return 'Dataset/jhu_crowd_v2.0/', '/images', '/ground_truth_npy'
        elif (dataset_name.lower().find('shtc') >= 0 or dataset_name.lower().find(
                'shanghai') >= 0) and dataset_name.lower().find('a') >= 0:
            return 'Dataset/ShanghaiTech/part_A_final/', '_data/images', '_data/ground_truth_npy'
        elif (dataset_name.lower().find('shtc') >= 0 or dataset_name.lower().find(
                'shanghai') >= 0) and dataset_name.lower().find('b') >= 0:
            return 'Dataset/ShanghaiTech/part_B_final/', '_data/images', '_data/ground_truth_npy'
        elif dataset_name.lower().find('qnrf') >= 0:
            return 'Dataset/UCF-QNRF/', '', '/ground_truth_npy'
        elif dataset_name.lower().find('nwpu') >= 0:
            return 'Dataset/NWPU/', '', '/ground_truth_npy'

    @staticmethod
    def get_name_from_no(dataset_name: str, set: str, prefix: str, img_suffix: str, dotmap_suffix: str, img_no):
        '''
        :param dataset_name:
        :param set: train, val, test
        :param prefix:
        :param img_suffix:
        :param dotmap_suffix:
        :param img_no: int or str, if it's str, please give the complete '*' in '*.jpg'
        :return:  give the number of the img & dot in the dataset, return the path of  the img & dot.
        '''
        if type(img_no) is int:
            if dataset_name.lower().startswith('jhu'):
                img_path = ''.join((prefix, set, img_suffix, '/', str(img_no).zfill(4), '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', str(img_no).zfill(4), '.npy'))
            elif dataset_name.lower().find('shtc') >= 0 or dataset_name.lower().find(
                    'shanghai') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/IMG_', str(img_no), '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/GT_IMG_', str(img_no), '.npy'))
            elif dataset_name.lower().find('qnrf') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/img_', str(img_no).zfill(4), '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/img_', str(img_no).zfill(4), '_ann', '.npy'))
            elif dataset_name.lower().find('nwpu') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/', str(img_no).zfill(4), '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', str(img_no).zfill(4), '.npy'))
        elif type(img_no) is str:
            if dataset_name.lower().startswith('jhu'):
                img_path = ''.join((prefix, set, img_suffix, '/', img_no, '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', img_no, '.npy'))
            elif dataset_name.lower().find('shtc') >= 0 or dataset_name.lower().find(
                    'shanghai') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/', img_no, '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', 'GT_', img_no, '.npy'))
            elif dataset_name.lower().find('qnrf') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/', img_no, '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', img_no, '_ann', '.npy'))
            elif dataset_name.lower().find('nwpu') >= 0:
                img_path = ''.join((prefix, set, img_suffix, '/', img_no, '.jpg'))
                dotmap_path = ''.join((prefix, set, dotmap_suffix, '/', img_no, '.npy'))
        else:
            raise ValueError('img_no is not int or str')

        return img_path, dotmap_path

    @staticmethod
    def get_no_from_name(dataset_name: str, name: str):
        '''

        Args:
            dataset_name ():
            name (): img_path or dot_path

        Returns: number string of images in *.jpg or *.npy  (for instance: 0027.jpg --> '0027', GT_IMG_10.npy --> '10')

        '''
        if dataset_name.lower().startswith('jhu'):
            no = os.path.split(name)[-1].split('.')[0]
        elif dataset_name.lower().find('shtc') >= 0 or dataset_name.lower().find(
                'shanghai') >= 0:
            no = os.path.split(name)[-1].split('.')[0].split('_')[-1]
        elif dataset_name.lower().find('qnrf') >= 0:
            no = os.path.split(name)[-1].split('.')[0].split('_')[1]
        elif dataset_name.lower().find('nwpu') >= 0:
            no = os.path.split(name)[-1].split('.')[0]

        return no

    @staticmethod
    def get_data(dataset_name: str, set: str, img_no, min_side_length=0, max_side_length=np.inf,
                 is_gray=False):
        '''

        Args:
            dataset_name ():
            set ():  train, test or val
            img_no (int or str): the serial number of image, if it's str, please give complete '*' in '*.jpg' . if it's
                                 int, just give the exact number, e.g. 0031.jpg --> 31 .
            min_side_length ():
            max_side_length ():  the shorter edge of image should in the range [min_side_length,max_side_lenght], if it
                                is not, then proportionally resize it to the range.

        Returns: img & dot map

        '''
        prefix, img_suffix, dotmap_suffix = Directory_path.prefix_suffix(dataset_name)
        img_path, dotmap_path = Directory_path.get_name_from_no(dataset_name, set, prefix, img_suffix, dotmap_suffix,
                                                                img_no)

        if is_gray:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        dot_ann = np.load(dotmap_path)
        if dot_ann.shape[0] > 0:
            if dot_ann.ndim == 1:  # if there is only 1 head in the image, we need to expand the dimension to 2.
                dot_ann = np.expand_dims(dot_ann[:2], axis=0)
            dot_ann = dot_ann[:, :2]
        else:
            dot_ann = dot_ann

        # resize the image & dotmap to make the shorter length fall into the range [min_side_length,max_side_lenght].
        w, h = img.size
        h, w, ratio = cal_new_size(h, w, min_side_length, max_side_length)
        if ratio != 1:
            try:
                img, dot_ann = idp.resize(img, dot_ann, np.array([w, h]))
            except ValueError as e:
                print(dot_ann.shape)
                print(dot_ann, img_no)
        return img, dot_ann


class Batch_image_dotmap_processing:
    @staticmethod
    def resize(dataset_name: str, set: str, min_side_length: int, max_side_length: int, dotmap_together=True,
               exception=(), is_gray=False):
        '''

        :param dataset_name:
        :param set:  train, val, test
        :param min_side_length:  shorter edge's min length
        :param max_side_length:  shorter edge's max length
        :param dotmap_together:  decide whether to resize the dotmap together
        :return: resized dataset replacing original dataset at the same location. if dotmap_together is true, the dataset's
                dotmap will only keep two columns, one column
                for x axis, one column for y axis, any other column will be discarded. but for jhu++, the third and fourth
                columns will be kept, they are head box's width and height.
        '''

        def resize_img(img_path, dotmap_path, dotmap_together, is_gray):
            if is_gray:
                img = Image.open(img_path).convert('L')
            else:
                img = Image.open(img_path).convert('RGB')

            if dotmap_together:
                dot_ann = np.load(dotmap_path)
                if dot_ann.shape[0] > 0:
                    if dot_ann.ndim == 1:
                        dot_ann = np.expand_dims(dot_ann, axis=0)
                    if img_path.lower().find('jhu') >= 0:
                        dot_ann = dot_ann[:, :4]
                    else:
                        dot_ann = dot_ann[:, :2]

            w, h = img.size
            h, w, ratio = cal_new_size(h, w, min_side_length, max_side_length)
            if ratio != 1:
                if dotmap_together:
                    img = img.resize(np.array([w, h]))
                    img.save(img_path, quality=95)
                    if dot_ann.shape[0] > 0:
                        dot_ann = dot_ann * ratio
                    np.save(dotmap_path, dot_ann)
                    print(img_path, dotmap_path)
                else:
                    img = img.resize(np.array([w, h]))
                    img.save(img_path, quality=95)
                    print(img_path)

        prefix, img_suffix, dotmap_suffix = Directory_path.prefix_suffix(dataset_name)

        for img_path in glob.glob(prefix + set + img_suffix + '/*.jpg'):
            num = Directory_path.get_no_from_name(dataset_name, img_path)
            if int(num) not in exception:
                img_path, dotmap_path = Directory_path.get_name_from_no(dataset_name, set, prefix, img_suffix,
                                                                        dotmap_suffix, int(num))
                resize_img(img_path, dotmap_path, dotmap_together, is_gray)

        for file in glob.glob(prefix + set.lower() + '/*.mat'):
            os.remove(file)


class Dataset_preparation:
    @staticmethod
    def SHTCA():
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC('Dataset/ShanghaiTech/part_A_final/train_data/ground_truth',
                                                  'Dataset/ShanghaiTech/part_A_final/train_data/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC('Dataset/ShanghaiTech/part_A_final/test_data/ground_truth',
                                                  'Dataset/ShanghaiTech/part_A_final/test_data/ground_truth_npy')

    @staticmethod
    def SHTCB():
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC('Dataset/ShanghaiTech/part_B_final/train_data/ground_truth',
                                                  'Dataset/ShanghaiTech/part_B_final/train_data/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_SHTC('Dataset/ShanghaiTech/part_B_final/test_data/ground_truth',
                                                  'Dataset/ShanghaiTech/part_B_final/test_data/ground_truth_npy')

    @staticmethod
    def QNRF():
        Trans_gt_to_ndarray.trans_ann_to_npy_QNRF('Dataset/UCF-QNRF/train', 'Dataset/UCF-QNRF/train/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_QNRF('Dataset/UCF-QNRF/test', 'Dataset/UCF-QNRF/test/ground_truth_npy')

    @staticmethod
    def JHU():
        Trans_gt_to_ndarray.trans_ann_to_npy_JHU('Dataset/jhu_crowd_v2.0/train/gt', 'Dataset/jhu_crowd_v2.0/train/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_JHU('Dataset/jhu_crowd_v2.0/val/gt', 'Dataset/jhu_crowd_v2.0/val/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_JHU('Dataset/jhu_crowd_v2.0/test/gt', 'Dataset/jhu_crowd_v2.0/test/ground_truth_npy')

        Batch_image_dotmap_processing.resize('JHU', 'train', 512, 2048)
        Batch_image_dotmap_processing.resize('JHU', 'val', 0, 2048)
        Batch_image_dotmap_processing.resize('JHU', 'test', 0, 2048)

    @staticmethod
    def NWPU():
        Trans_gt_to_ndarray.trans_ann_to_npy_NWPU('Dataset/NWPU/train', 'Dataset/NWPU/train/ground_truth_npy')
        Trans_gt_to_ndarray.trans_ann_to_npy_NWPU('Dataset/NWPU/val', 'Dataset/NWPU/val/ground_truth_npy')

        Batch_image_dotmap_processing.resize('NWPU', 'train', 512, 2048)
        Batch_image_dotmap_processing.resize('NWPU', 'val', 0, 2048, exception=(3234,))
        exception_set = [i for i in range(3110, 3610) if i != 3234]
        Batch_image_dotmap_processing.resize('NWPU', 'val', 0, 3024, exception=exception_set)


