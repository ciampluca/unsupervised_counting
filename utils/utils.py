from PIL import Image
import numpy as np

import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import PairedCrop, RandomHorizontalFlip, CustomResize


def get_transforms(general_transforms=None, img_transforms=None, target_transforms=None, train=None, dim_resize=None):

    transforms_list = []

    if general_transforms:
        if dim_resize:
            transforms_list.append(CustomResize(dim=dim_resize))
        if train:
            transforms_list.append(RandomHorizontalFlip())
        transforms_list.append(PairedCrop())
    if img_transforms:
        transforms_list.append(ToTensor())
        transforms_list.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225]))
    if target_transforms:
        transforms_list.append(ToTensor())

    return Compose(transforms_list)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_discriminator_accuracy(label, pred, cfg):
    boolean_label = label.type(torch.BoolTensor).to(cfg.device)
    acc = torch.mean((torch.eq(torch.sigmoid(pred) > .5, boolean_label)).type(torch.FloatTensor))

    return acc
