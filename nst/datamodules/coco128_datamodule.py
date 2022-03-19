from typing import Optional
import os
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder


class COCO128DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=1, image_size=(256, 256)):
        super().__init__()
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize(image_size),
            ]
        )

        self.batch_size = batch_size
        self.save_hyperparameters()

    def prepare_data(self):
        download_and_extract_archive(
            "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip",
            download_root=self.data_dir,
        )

    def setup(self, stage=None):
        data_path = os.path.join([self.data_dir, "coco128", "images"])
        coco_dataset = ImageFolder(data_path, self.transform)
        coco_dataset_split = random_split(coco_dataset, [120, 8])
        self.train_dataset, self.val_dataset = coco_dataset_split[0], coco_dataset_split[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
