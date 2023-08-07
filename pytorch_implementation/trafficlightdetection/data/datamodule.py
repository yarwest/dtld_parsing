import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd

from trafficlightdetection.data.dataset import LTLDDataset
from trafficlightdetection.data.split import MultiLabelStratifiedSplitter
from trafficlightdetection.data.transforms import (
    get_test_transforms,
    get_training_transforms,
)


def collate_fn(batch):
    return tuple(zip(*batch))

class LTLDDataModule(pl.LightningDataModule):
    """
    Class describing the DriveU Dataset containing a list of images

    Attributes:
        images (List of DriveuImage)  All images of the dataset
        train_data_path (string):           Path of the dataset (.json)
        train_data_path (string):           Path of the dataset (.json)
    """
    def __init__(
        self,
        train_label_path,
        test_label_path,
        batch_size=4,
        num_workers=1,
        random_state=None,
    ):
        super().__init__()
        self.images = []
        self.train_label_path = train_label_path
        self.test_label_path = test_label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_set = LTLDDataset(
            self.train_label_path,
            get_training_transforms(),
        )
        original_train_dataset = LTLDDataset(
            self.train_label_path,
            get_test_transforms(),
        )
        
        self.test_set = LTLDDataset(
            self.test_label_path,
            get_test_transforms(),
        )

        # splitter = MultiLabelStratifiedSplitter(
        #     original_train_dataset.images,
        #     "image_path",
        #     "state",
        #     random_state=self.random_state,
        # )
        # splitter.split()

        # self.valid_set = torch.utils.data.Subset(
        #     original_train_dataset, splitter.valid_indices()
        # )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.valid_set,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         collate_fn=collate_fn,
    #     )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
