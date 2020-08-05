import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, normalize

from utils.utils import get_transforms


class NDISPark(Dataset):

    def __init__(self, root_dataset, phase="source", transform=None, img_transform=None, target_transform=None):
        assert phase == "source" or phase == "target" or phase == "test", "phase not present"

        self.imgs_path = os.path.join(root_dataset, phase + '_data/images')
        self.densities_path = os.path.join(root_dataset, phase + '_data/densitymaps')
        self.data_files = [filename for filename in os.listdir(self.imgs_path)
                           if os.path.isfile(os.path.join(self.imgs_path, filename))]
        self.transform = transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.phase = phase

        # We just need number of vehicles present in the images
        if phase == "test":
            self.gt = {}
            gt_txt_path = os.path.join(root_dataset, phase + "_data", "test_counting_gt.txt")
            with open(gt_txt_path) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            content = content[:-1]
            for line in content:
                (key, val) = line.split()
                self.gt[key] = float(val)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        index = index % len(self.data_files)
        fname = self.data_files[index]

        # Loading image
        img = Image.open(os.path.join(self.imgs_path, fname))
        if img.mode == 'L' or img.mode == 'RGBA':
            img = img.convert('RGB')

        # Loading density map. If we are in the test phase we just need the total number of vehicles, so density
        # maps are just fake black images
        if self.phase == "test":
            den_map = Image.new('F', img.size)
        else:
            den_map = Image.open(os.path.join(self.densities_path, fname.rsplit(".", 1)[0] + ".tiff"))

        if self.transform is not None:
            img, den_map = self.transform((img, den_map))
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            den_map = self.target_transform(den_map)

        if self.phase == "test":
            # Retrieving gt number of vehicles
            key = fname.rsplit(".", 1)[0]
            num = self.gt.get(key)
            return {'image': img, 'densitymap': den_map, 'name': fname, 'num': num}
        else:
            return {'image': img, 'densitymap': den_map, 'name': fname}


# # Testing code
# if __name__ == "__main__":
#     root = "/media/luca/Dati_2_SSD/datasets/vehicles_counting/NDISPark"
#     root_val = "/media/luca/Dati_2_SSD/datasets/vehicles_counting/NDISPark"
#     phase = "target"
#     DIM_RESIZE = None
#
#     train_dataset = NDISPark(
#         root_dataset=root,
#         transform=get_transforms(general_transforms=True, train=True, dim_resize=DIM_RESIZE),
#         img_transform=get_transforms(img_transforms=True),
#         target_transform=get_transforms(target_transforms=True),
#     )
#     val_dataset = NDISPark(
#         root_dataset=root_val,
#         phase=phase,
#         transform=get_transforms(general_transforms=True, dim_resize=DIM_RESIZE),
#         img_transform=get_transforms(img_transforms=True,),
#         target_transform=get_transforms(target_transforms=True),
#     )
#
#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=False,
#         batch_size=1,
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         shuffle=False,
#         batch_size=1,
#         num_workers=1,
#     )
#
#     for i, data in enumerate(train_dataloader):
#         name = data['name'][0].rsplit(".", 1)[0]
#         print(name)
#
#         image = data['image'].squeeze(dim=0)
#         image = normalize(image, mean=[-0.5 / 0.225, -0.5 / 0.225, -0.5 / 0.225], std=[1 / 0.225, 1 / 0.225, 1 / 0.225])
#         pil_image = to_pil_image(image)
#         pil_image.save(os.path.join("../output_debug/", name + ".png"))
#
#         if phase == "test":
#             num = data['num'].cpu().item()
#             print(num)
#         else:
#             density_map = data['densitymap'].squeeze(dim=0)
#             pil_density_map = to_pil_image((density_map/torch.max(density_map))*255)
#             np_density_map = density_map.cpu().detach().numpy().astype(np.float32)
#             unique, counts = np.unique(np_density_map, return_counts=True)
#             num = np.sum(np_density_map)
#             num_double_check = len(unique)-1
#             pil_density_map.save(os.path.join("../output_debug/", "density_" + name + ".tiff"))
#             print(num)
#             print(num_double_check)



