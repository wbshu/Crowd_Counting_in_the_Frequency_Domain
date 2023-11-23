import numpy as np
import torch
from PIL import Image


class ImgTensor_dotTensor_processing():
    @staticmethod
    def crop(img_tensor, dot_tensor, crop_position):
        '''
        It can serve the RGB and Gray image tensor.
        Args:
            img_tensor ():
            dot_tensor ():
            crop_position (4-int-tuple): image axis: (left, upper, right, lower) , img.crop(crop_position).ToTensor() ==
                                        crop(img.ToTensor(), dot_tensor, crop_position)[0]

        Returns: cropped image tensor together with its dot_tensor

        '''
        assert (0 <= crop_position[0]) & (crop_position[0] < crop_position[2]) & (
                crop_position[2] <= img_tensor.size()[2]) & (0 <= crop_position[1]) & (
                       crop_position[1] < crop_position[3]) & (crop_position[3] <= img_tensor.size()[1])

        img_tensor = img_tensor[:, crop_position[1]:crop_position[3], crop_position[0]:crop_position[2]]

        if dot_tensor.shape[0] > 0:
            mask = (dot_tensor[:, 0] > crop_position[0]) & (dot_tensor[:, 0] < crop_position[2]) & (
                    dot_tensor[:, 1] > crop_position[1]) & (dot_tensor[:, 1] < crop_position[3])
            dot_tensor = dot_tensor[mask] - torch.tensor(crop_position[0:2]).to(dtype=dot_tensor.dtype,
                                                                                device=dot_tensor.device)

        return img_tensor, dot_tensor

    @staticmethod
    def random_crop(img_tensor, dot_tensor, size, crop_mode):
        '''
        random crop
        Args:
            img_tensor ():
            dot_tensor ():
            size (single value or 2-arraylike): crop size, single value: cropped to (size,size); 2-arraylike: cropped to
                                                 size (w,h).
            crop_mode (callable crop function):  crop function

        Returns: the image tensor and dotted_map tensor after random cropping
        '''
        selectable_range = np.array([img_tensor.shape[-1], img_tensor.shape[-2]]) - np.array(size)
        assert (selectable_range >= 0).all()
        left_up = np.floor(np.random.rand(2) * selectable_range).astype(int)
        right_down = np.floor(left_up + size).astype(int)
        return crop_mode(img_tensor, dot_tensor, tuple(np.concatenate((left_up, right_down))))

    @staticmethod
    def random_mirrow(img_tensor, dot_tensor):
        '''
        Random mirrow the image
        Args:
            img_tensor ():
            dot_tensor (n×2 tensor): first column x position, second column y position. In Image coordinates.

        Returns: random mirrowed image tensor and dot tensor.

        '''
        if np.random.rand(1) > 0.5:
            img_tensor = torch.flip(img_tensor, [-1])
            if dot_tensor.shape[0] > 0:
                dot_tensor[:, 0] = img_tensor.shape[-1] - dot_tensor[:, 0]
        return img_tensor, dot_tensor


class Image_dotmap_processing():
    @staticmethod
    def crop(img, dotted_map, crop_position=(0, 0, 512, 512)):
        '''
        General crop
        Args:
            img (Image): PIL Image
            dotted_map (n × 2 ndarray):
            crop_position (4-tuple): (left, upper, right, lower)

        Returns: the image and dotted_map after cropping

        '''
        img = img.crop(crop_position)
        if dotted_map.shape[0] > 0:
            mask = (dotted_map[:, 0] > crop_position[0]) & (dotted_map[:, 0] < crop_position[2]) & (
                    dotted_map[:, 1] > crop_position[1]) & (dotted_map[:, 1] < crop_position[3])
            dotted_map = dotted_map[mask]
            dotted_map[:, 0:2] = dotted_map[:, 0:2] - crop_position[0:2]

        return img, dotted_map

    @staticmethod
    def random_crop(img, dotted_map, size):
        '''
        random crop
        Args:
            img (Image): PIL Image
            dotted_map (n×2 ndarray): head position
            size (single value or 2-arraylike): crop size, single value: cropped to (size,size); 2-arraylike: cropped to
                                                 size (w,h).

        Returns: the image and dotted_map after random cropping

        '''
        selectable_range = np.array(img.size) - np.array(size)
        assert (selectable_range >= 0).all()
        left_up = np.random.rand(2) * selectable_range
        right_down = left_up + size
        return Image_dotmap_processing.crop(img, dotted_map, tuple(np.concatenate((left_up, right_down))))

    @staticmethod
    def resize(img, dotted_map, size=512):
        '''
        Resize the image and dotted map
        Args:
            img (Image): PIL Image
            dotted_map ((n×2 ndarray): head position
            size (single value or 2-arraylike): resizing size, single value: resize to (size,size); 2-arraylike: resize
                                                to size (w,h)

        Returns: image and dotted map after resizing

        '''
        size = np.array(size)
        ratio = size / np.array(img.size)
        if size.size == 1:
            image = img.resize((size, size))
        elif size.size == 2:
            image = img.resize(size)

        if dotted_map.shape[0] > 0:
            dotted_map = dotted_map * ratio

        return image, dotted_map

    @staticmethod
    def random_mirrow(img, dotted_map):
        '''
        Random mirrow the image
        Args:
            img (Image): PIL Image
            dotted_map (ndarray): head position

        Returns: random mirrowed image and dotted map

        '''
        w, h = img.size
        if np.random.rand(1) > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if dotted_map.shape[0] > 0:
                dotted_map[:, 0] = w - dotted_map[:, 0]
        return img, dotted_map


class Generating_data_from_dotted_annotation:
    @staticmethod
    def construct_characteristic_function(head_position, bandwidth, origin=0,
                                          step=30, step_length=0.01):
        '''

        discretized characteristic function

        Args:
            head_position (n×2 tensor): 2D tensor, the length of first dimension corresponds to the number of heads,
            each line corresponds to a head, the first column is the x coordinate, the second column is the y coordinate.

            bandwidth (1D tensor or single value): the bandwidth of each head's Gauss

            origin (1×2 tensor or single value): decide the origin of the image plane

            step (single integer): how many steps the c.f. span in the plane (originate in the origin and expand to
            4 directions, up, down, left, right)

            step_length (single value): the span of each step, the final presented domain of the c.f. is decided by step
            and step_length

        Returns: the c.f. values in the range [-step * step_length, step * step_length) × [-step * step_length, step * step_length)

        '''
        if head_position.shape[0] > 0:
            gauss_mean = head_position - origin
            if not isinstance(bandwidth, (int, float)):
                bandwidth = torch.reshape(bandwidth, (1, 1, head_position.shape[0])).to(dtype=gauss_mean.dtype,
                                                                                        device=gauss_mean.device)
            plane = torch.cat(
                [torch.arange(-step, step).unsqueeze(0).expand(2 * step, 2 * step).unsqueeze(2) * step_length,
                 torch.arange(-step, step).unsqueeze(1).expand(2 * step, 2 * step).unsqueeze(2) * step_length],
                dim=2).to(device=gauss_mean.device)
            angle = torch.matmul(plane.to(dtype=gauss_mean.dtype), gauss_mean.t())
            length = torch.exp(-1 / 2 * (plane * plane).sum(dim=2, keepdim=True) * bandwidth ** 2)
            angle_real = torch.cos(angle)
            angle_img = torch.sin(angle)

            cf_real = (angle_real * length).sum(dim=2, keepdim=True)
            cf_img = (angle_img * length).sum(dim=2, keepdim=True)
            return torch.cat([cf_real, cf_img], dim=2).to(dtype=torch.float32)
        else:    # for zero-people ground truth
            return torch.zeros(step * 2, step * 2, 2).to(dtype=torch.float32)


