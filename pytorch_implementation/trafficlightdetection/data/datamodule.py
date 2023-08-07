import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from trafficlightdetection.data.dataset import LisaTrafficLightDataset
from trafficlightdetection.data.split import MultiLabelStratifiedSplitter
from trafficlightdetection.data.transforms import (
    get_test_transforms,
    get_training_transforms,
)


def collate_fn(batch):
    return tuple(zip(*batch))


class LisaTrafficLightDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        train_size=0.7,
        valid_size=0.15,
        test_size=0.15,
        time_of_day="both",
        annotation_type="BOX",
        batch_size=4,
        num_workers=1,
        random_state=None,
    ):
        if train_size + valid_size + test_size != 1.0:
            raise ValueError("Set sizes must sum up to 1.0")

        super().__init__()
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.time_of_day = time_of_day
        self.annotation_type = annotation_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state

    def prepare_data(self):
        pass

    def setup(self, stage):
        transformed_dataset = LisaTrafficLightDataset(
            self.dataset_path,
            get_training_transforms(),
            time_of_day=self.time_of_day,
            annotation_type=self.annotation_type,
        )
        original_dataset = LisaTrafficLightDataset(
            self.dataset_path,
            get_test_transforms(),
            time_of_day=self.time_of_day,
            annotation_type=self.annotation_type,
        )

        annotations = transformed_dataset.annotations
        splitter = MultiLabelStratifiedSplitter(
            annotations,
            "filename",
            "class",
            train_size=self.train_size,
            valid_size=self.valid_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        splitter.split()

        self.train_set = torch.utils.data.Subset(
            transformed_dataset, splitter.train_indices()
        )
        self.valid_set = torch.utils.data.Subset(
            original_dataset, splitter.valid_indices()
        )
        self.test_set = torch.utils.data.Subset(
            original_dataset, splitter.test_indices()
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
