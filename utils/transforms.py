import random
from PIL import Image
import numpy as np

import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):

    def __call__(self, img_and_density):
        """
        img: PIL.Image
        img_and_density: PIL.Image
        """
        img, density_map = img_and_density

        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), density_map.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img, density_map


class PairedCrop(object):
    """
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the neural network,
    we must promise that the size of input image is the corresponding factor.
    """

    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_density):
        """
        img_and_density: PIL.Image
        """
        img, density_map = img_and_density

        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        density_map = F.crop(density_map, i, j, th, tw)

        return img, density_map


class CustomResize(object):

    def __init__(self, dim=480):
        self.dim = dim

    def __call__(self, img_and_density):
        img, density_map = img_and_density
        np_den_map = np.array(density_map)
        num_objs = np.sum(np_den_map)

        img = F.resize(img, size=self.dim, interpolation=Image.ANTIALIAS)
        density_map = F.resize(density_map, size=self.dim, interpolation=Image.NEAREST)

        # Ensure that sum=#objects after resizing
        np_den_map = np.array(density_map)
        if np.sum(np_den_map) != 0.0:
            np_den_map = num_objs * np_den_map / np.sum(np_den_map)
        density_map = Image.fromarray(np_den_map, mode="F")

        return img, density_map
