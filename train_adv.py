import sys
import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms.functional import normalize

from config import Config
from models.CSRNet import CSRNet
from models.discriminator import FCDiscriminator
from utils.utils import random_seed, get_transforms, compute_discriminator_accuracy
from datasets.NDISPark import NDISPark

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Config default values
EPOCHS = 300
BATCH_SIZE = 1
ROOT_DATASET = "/media/luca/Dati_2_SSD/datasets/vehicles_counting/NDISPark"
ROOT_VAL_DATASET = "/media/luca/Dati_2_SSD/datasets/vehicles_counting/NDISPark"
LAMBDA_ADV_LOSS = 5e-05
LAMBDA_DISC_LOSS = 0.0001


def main(args):
    print(args)

    # Loading configuration
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        root_dataset=args.source_dataset_path,
        root_val_dataset=args.target_dataset_path,
        lambda_adv_loss=args.lambda_adv,
        lambda_disc_loss=args.lambda_disc,
    )

    # Reproducibility
    seed = cfg.seed
    if torch.cuda.is_available():
        random_seed(seed, True)
    else:
        random_seed(seed, False)

    # Defining exp name
    exp_name = "_Train{}_Val{}_{}_advLoss{}_discLoss{}_lr{}_batchSize{}".\
        format(cfg.root_dataset.rsplit("/", 1)[1], cfg.root_val_dataset.rsplit("/", 1)[1], cfg.model_name,
               cfg.lambda_adv_loss, cfg.lambda_disc_loss, cfg.lr_base, cfg.batch_size)

    # Creating tensorboard writer
    tensorboard_writer = SummaryWriter(comment=exp_name)

    # Loading model
    model = CSRNet().to(cfg.device)

    # Loading discriminator
    discriminator = FCDiscriminator(num_classes=1).to(cfg.device)

    # Defining criterion and optimizer for the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr_base,
    )

    # Defining criterion and optimizer for the discriminator
    discriminator_criterion = nn.BCEWithLogitsLoss()
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=cfg.discriminator_lr_base,
        betas=(0.9, 0.99),
    )

    # Creating datasets
    train_dataset = NDISPark(
        root_dataset=cfg.root_dataset,
        phase="source",
        transform=get_transforms(general_transforms=True, train=True),
        img_transform=get_transforms(img_transforms=True),
        target_transform=get_transforms(target_transforms=True),
    )
    val_dataset = NDISPark(
        root_dataset=cfg.root_val_dataset,
        phase="target",
        transform=get_transforms(general_transforms=True),
        img_transform=get_transforms(img_transforms=True),
        target_transform=get_transforms(target_transforms=True),
    )
    target_dataset = NDISPark(
        root_dataset=cfg.root_val_dataset,
        phase="target",
        transform=get_transforms(general_transforms=True),
        img_transform=get_transforms(img_transforms=True),
        target_transform=get_transforms(target_transforms=True),
    )

    # Creating samplers for target dataloader
    weights = [1.0] * len(target_dataset)
    target_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Creating dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=cfg.batch_size,
        sampler=target_sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Defining labels for adversarial training
    source_label = 0
    target_label = 1

    min_mae, min_mse, min_are = sys.maxsize, sys.maxsize, sys.maxsize
    min_mae_epoch, min_mse_epoch, min_are_epoch = -1, -1, -1
    # Iterating over epochs...
    for epoch in range(1, cfg.epochs):
        model.train()
        discriminator.train()
        epoch_loss, disc_epoch_loss, model_epoch_loss, adv_epoch_loss = 0.0, 0.0, 0.0, 0.0
        epoch_mae, epoch_mse, epoch_are = 0.0, 0.0, 0.0
        epoch_disc_adv_acc, epoch_disc_1_acc, epoch_disc_2_acc = 0.0, 0.0, 0.0

        # Creating an iterator over the target dataloader
        target_iterator = iter(target_dataloader)

        # Training for one epoch
        for i, source_data in enumerate(tqdm.tqdm(train_dataloader)):
            # Setting grads to zero
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            ######################
            # Training the model #
            ######################

            # Don't accumulate grads in Discriminator
            for param in discriminator.parameters():
                param.requires_grad = False

            # TRAINING WITH SOURCE LABELED IMAGE
            # Retrieving source image and gt
            source_image = source_data['image'].to(cfg.device)
            source_gt_density_map = source_data['densitymap'].to(cfg.device)
            # Computing pred density map
            source_pred_density_map = model(source_image)
            # Computing loss
            source_loss = criterion(source_pred_density_map, source_gt_density_map)
            source_loss.backward()
            model_epoch_loss += source_loss.item()

            # Computing MAE, MSE and ARE
            mae = abs(source_pred_density_map.data.sum() - source_gt_density_map.data.sum())
            epoch_mae += mae.item()
            mse = (source_pred_density_map.data.sum() - source_gt_density_map.data.sum()) ** 2
            epoch_mse += mse.item()
            are = abs(source_pred_density_map.data.sum() - source_gt_density_map.data.sum()) / torch.clamp(
                source_gt_density_map.data.sum(), min=1)
            epoch_are += are.item()

            # TRAINING WITH TARGET UNLABELED IMAGE (ADV LOSS)
            # Retrieving target image
            target_data = target_iterator.__next__()
            target_image = target_data['image'].to(cfg.device)
            # Computing pred density map
            target_pred_density_map = model(target_image)
            # Computing output of the discriminator
            discriminator_pred = discriminator(target_pred_density_map)
            # Computing adv loss (between discriminator prediction and source-values label)
            source_values_label = torch.FloatTensor(discriminator_pred.data.size()).fill_(source_label).to(cfg.device)
            adv_loss = discriminator_criterion(discriminator_pred, source_values_label)
            adv_loss = cfg.lambda_adv_loss * adv_loss
            adv_loss.backward()
            adv_epoch_loss += adv_loss.item()

            # Computing accuracy of the discriminator
            disc_adv_acc = compute_discriminator_accuracy(source_values_label, discriminator_pred, cfg)
            epoch_disc_adv_acc += disc_adv_acc

            # Computing total loss and backwarding it
            loss = source_loss + adv_loss
            epoch_loss += loss.item()
            # loss.backward()
            optimizer.step()

            ##############################
            # Training the discriminator #
            ##############################

            # Bringing back requires_grad
            for param in discriminator.parameters():
                param.requires_grad = True

            # TRAINING WITH SOURCE LABELED IMAGE
            # Computing output of the discriminator
            source_pred_density_map = source_pred_density_map.detach()
            discriminator_pred = discriminator(source_pred_density_map)
            # Computing discriminator loss (between discriminator prediction and source-values label)
            source_values_label = torch.FloatTensor(discriminator_pred.data.size()).fill_(source_label).to(cfg.device)
            disc_loss = discriminator_criterion(discriminator_pred, source_values_label)
            disc_loss = cfg.lambda_disc_loss * disc_loss
            # Computing accuracy of the discriminator
            disc_1_acc = compute_discriminator_accuracy(source_values_label, discriminator_pred, cfg)
            epoch_disc_1_acc += disc_1_acc
            # Backwarding loss
            disc_epoch_loss += disc_loss.item()
            disc_loss.backward()

            # TRAINING WITH TARGET UNLABELED IMAGE
            # Computing output of the discriminator
            target_pred_density_map = target_pred_density_map.detach()
            discriminator_pred = discriminator(target_pred_density_map)
            # Computing discriminator loss (between discriminator prediction and target-values label)
            target_values_label = torch.FloatTensor(discriminator_pred.data.size()).fill_(target_label).to(cfg.device)
            disc_loss = discriminator_criterion(discriminator_pred, target_values_label)
            disc_loss = cfg.lambda_disc_loss * disc_loss
            # Computing accuracy of the discriminator
            disc_2_acc = compute_discriminator_accuracy(target_values_label, discriminator_pred, cfg)
            epoch_disc_2_acc += disc_2_acc
            # Backwarding loss
            disc_epoch_loss += disc_loss.item()
            disc_loss.backward()

            # Performing optimizer step
            discriminator_optimizer.step()

        tensorboard_writer.add_scalar('Train/Loss', epoch_loss / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Disc_Loss', disc_epoch_loss / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/MAE', epoch_mae / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/MSE', epoch_mse / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/ARE', epoch_are / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Discr_Adv_Acc', epoch_disc_adv_acc / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Discr_1_Acc', epoch_disc_1_acc / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Discr_2_Acc', epoch_disc_2_acc / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Model_Loss', model_epoch_loss / len(train_dataset), epoch)
        tensorboard_writer.add_scalar('Train/Adv_Loss', adv_epoch_loss / len(train_dataset), epoch)

        # Validate the epoch
        model.eval()
        with torch.no_grad():
            epoch_mae, epoch_mse, epoch_are, epoch_loss = 0.0, 0.0, 0.0, 0.0

            for i, data in enumerate(tqdm.tqdm(val_dataloader)):
                # Retrieving image and density map
                image = data['image'].to(cfg.device)
                gt_density_map = data['densitymap'].to(cfg.device)

                # Computing output and val loss
                pred_density_map = model(image)
                val_loss = criterion(pred_density_map, gt_density_map)
                epoch_loss += val_loss.item()
                pred_density_map = pred_density_map.detach()

                # Computing MAE and MSE
                mae = abs(pred_density_map.data.sum() - gt_density_map.data.sum())
                epoch_mae += mae.item()
                mse = (pred_density_map.data.sum() - gt_density_map.data.sum()) ** 2
                epoch_mse += mse.item()
                are = abs(pred_density_map.data.sum() - gt_density_map.data.sum()) / torch.clamp(
                    gt_density_map.data.sum(), min=1)
                epoch_are += are.item()

            epoch_mae /= len(val_dataset)
            epoch_mse /= len(val_dataset)
            epoch_are /= len(val_dataset)
            epoch_loss /= len(val_dataset)

            # Saving last model
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_folder, "last.pth"))
            # Eventually saving best models
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_folder, str(epoch) + "_mae.pth"))
            if epoch_mse < min_mse:
                min_mse, min_mse_epoch = epoch_mse, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_folder, str(epoch) + "_mse.pth"))
            if epoch_are < min_are:
                min_are, min_are_epoch = epoch_are, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoint_folder, str(epoch) + "_are.pth"))
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch,
                  min_mae_epoch, 'MSE: ', epoch_mse, 'ARE: ', epoch_are)

            tensorboard_writer.add_scalar('Val/MAE', epoch_mae, epoch)
            tensorboard_writer.add_scalar('Val/MSE', epoch_mse, epoch)
            tensorboard_writer.add_scalar('Val/ARE', epoch_are, epoch)
            tensorboard_writer.add_scalar('Val/Loss', epoch_loss, epoch)
            tensorboard_writer.add_image(str(epoch) + '/Image',
                                         normalize(image.cpu().squeeze(dim=0),
                                                   mean=[-0.5 / 0.225, -0.5 / 0.225, -0.5 / 0.225],
                                                   std=[1 / 0.225, 1 / 0.225, 1 / 0.225]))
            tensorboard_writer.add_image(
                str(epoch) + '/Pred Count:' + str('%.2f' % (pred_density_map.cpu().squeeze(dim=0).sum())),
                torch.abs(pred_density_map.squeeze(dim=0)) / torch.max(pred_density_map.squeeze(dim=0)))
            tensorboard_writer.add_image(
                str(epoch) + '/GT count:' + str('%.2f' % (gt_density_map.cpu().squeeze(dim=0).sum())),
                gt_density_map.squeeze(dim=0) / torch.max(gt_density_map.squeeze(dim=0)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--source-dataset-path', default=ROOT_DATASET, help='source dataset root path')
    parser.add_argument('--target-dataset-path', default=ROOT_VAL_DATASET, help='target dataset root path')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int, help='batch_size')
    parser.add_argument('--lambda-adv', default=LAMBDA_ADV_LOSS, type=float, help='lambda for the adv loss')
    parser.add_argument('--lambda-disc', default=LAMBDA_DISC_LOSS, type=float, help='lambda for the discriminator loss')

    args = parser.parse_args()

    main(args)
