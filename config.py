import os
import time

import torch


class Config:

    def __init__(self, batch_size=None, lr_base=None, discriminator_lr_base=None, epochs=None, root_dataset=None,
                 root_val_dataset=None, checkpoint_folder=None, momentum=0.9, weight_decay=0.0001, model_name=None,
                 input_dim_resize=480, num_workers=4, lambda_adv_loss=0, lambda_disc_loss=0, dataset_random_split=None,
                 dataset_roi_masked=None, seed=10):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = 1 if not batch_size else batch_size
        self.lr_base = 1e-5 if not lr_base else lr_base
        self.discriminator_lr_base = 1e-5 if not discriminator_lr_base else discriminator_lr_base
        self.momentum = momentum
        self.dataset_random_split = dataset_random_split
        self.dataset_roi_masked = dataset_roi_masked
        self.weight_decay = weight_decay
        self.lambda_adv_loss = lambda_adv_loss
        self.lambda_disc_loss = lambda_disc_loss
        self.input_dim_resize = input_dim_resize
        self.epochs = 100 if not epochs else epochs
        self.num_workers = num_workers
        self.root_dataset = './data/NDISPark' if not root_dataset else root_dataset
        self.root_val_dataset = './data/NDISPark' if not root_val_dataset else root_val_dataset
        self.dataset_name = self.root_dataset.rsplit("/", 1)[1]
        self.model_name = 'CSRNet' if not model_name else model_name
        self.date_and_time = time.strftime("%Y%m%d%H%M")
        self.checkpoint_folder = os.path.join('./checkpoints', self.dataset_name, self.model_name, self.date_and_time) \
            if not checkpoint_folder else checkpoint_folder
        self.seed = seed

        os.makedirs(self.checkpoint_folder, exist_ok=True)

