import os
import tqdm
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.CSRNet import CSRNet
from datasets.NDISPark import NDISPark
from utils.utils import get_transforms


# Parameters
ROOT_DATASET = "/media/luca/Dati_2_SSD/datasets/vehicles_counting/NDISPark"
PHASE = "test"
MODEL_NAME = "CSRNet"
MODEL_CHECKPOINT = "/home/luca/workspace/unsupervised_counting/checkpoints/NDISPark/CSRNet/202008041749/74_mae.pth"
GT_TXT_FILE = True
RESULTS = "/home/luca/Downloads/temp_results/NDISPark/basic/results"
PREDS = "/home/luca/Downloads/temp_results/NDISPark/basic/preds"


def main():
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Creating output folder
    preds_output_folder = os.path.join(PREDS, "obtained_with_best_model_mae")
    if not os.path.exists(preds_output_folder):
        os.makedirs(preds_output_folder)

    # Loading model
    model = CSRNet()

    # Loading checkpoint
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))
    model.to(device)
    model.eval()

    dataset = NDISPark(
        root_dataset=ROOT_DATASET,
        phase=PHASE,
        transform=get_transforms(general_transforms=True),
        img_transform=get_transforms(img_transforms=True),
        target_transform=get_transforms(target_transforms=True),
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )

    total_mae, total_mse, total_are = 0.0, 0.0, 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            # Retrieving image and density map
            image = data['image'].to(device)
            gt_density_map = data['densitymap'].to(device)

            # Computing pred density map
            pred_density_map = model(image)

            # Computing MAE, MSE and ARE
            if GT_TXT_FILE:
                gt_num = data['num'].cpu().item()
                mae = abs(pred_density_map.data.sum() - gt_num)
                total_mae += mae.item()
                mse = (pred_density_map.data.sum() - gt_num) ** 2
                total_mse += mse.item()
                are = abs(pred_density_map.data.sum() - gt_num) / gt_num
                total_are += are.item()
            else:
                mae = abs(pred_density_map.data.sum() - gt_density_map.data.sum())
                total_mae += mae.item()
                mse = (pred_density_map.data.sum() - gt_density_map.data.sum()) ** 2
                total_mse += mse.item()
                are = abs(pred_density_map.data.sum() - gt_density_map.data.sum()) / torch.clamp(
                    gt_density_map.data.sum(), min=1)
                total_are += are.item()

            density_to_save = pred_density_map.detach()
            density_to_save = density_to_save.squeeze(0).squeeze(0).cpu().numpy()

            density_to_save = np.absolute(density_to_save)
            density_to_save = 255 * (density_to_save / np.max(density_to_save))
            density_to_save = density_to_save.astype(np.uint8)
            # density_to_save = (255 * (density_to_save - np.min(density_to_save)) / (
            #         np.max(density_to_save) - np.min(density_to_save))).astype(np.uint8)
            pil_density = Image.fromarray(density_to_save)
            pil_density.save(os.path.join(preds_output_folder, data['name'][0].rsplit(".", 1)[0] + ".png"))
            # pil_density.save(os.path.join(preds_output_folder, data['name'][0].rsplit(".", 1)[0] + ".tiff"))

            print("Image: {}, AE: {}, SE: {}, RE: {}".format(data['name'][0], mae.item(), mse.item(), are.item()))

    string_to_write = "Model: {}, Checkpoint: {}, MAE: {}, MSE: {}, ARE: {}".\
        format(MODEL_NAME, MODEL_CHECKPOINT, total_mae/len(dataset), total_mse/len(dataset), total_are/len(dataset))
    with open(os.path.join(RESULTS, "obtained_with_best_model_mae.txt"), "w") as result_file:
        result_file.write(string_to_write)
    print(string_to_write)


if __name__ == "__main__":
    main()
