import os

import cv2
import pandas as pd
import torch
import torchvision
import pytorch_lightning as pl

from trafficlightdetection.data.utils import LisaTrafficLightUtils


class LisaTrafficLightDataset(torch.utils.data.Dataset):
    # Im mniejszy numer, tym bardziej liczna klasa
    CLS_TO_INT = {
        "go": 1,
        "stop": 2,
        "stopLeft": 3,
        "warning": 4,
        "goLeft": 5,
        "warningLeft": 6,
        "goForward": 7,
    }

    INT_TO_CLS = {v: k for k, v in CLS_TO_INT.items()}

    def __init__(self, root, transforms, *, time_of_day="both", annotation_type="BOX"):
        if time_of_day not in ["day", "night", "both"]:
            raise ValueError("Invalid time of day value.")

        if annotation_type not in ["BOX", "BULB"]:
            raise ValueError("Invalid annotation type.")

        self.root = root
        self.transforms = transforms

        lisa_utils = LisaTrafficLightUtils(self.root, annotations_type=annotation_type)
        self.annotations = lisa_utils.get_annotations(time_of_day)
        self.images = pd.Series(self.annotations["filename"].unique())

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_name = os.path.basename(str(image_path))

        image = cv2.imread(os.path.join(self.root, str(image_path)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32")
        image /= 255

        image_annotations = self.annotations.loc[
            self.annotations["filename"] == image_path
        ]

        boxes = []
        labels = []
        for _, row in image_annotations.iterrows():
            xmin = row["min_x"]
            ymin = row["min_y"]
            xmax = row["max_x"]
            ymax = row["max_y"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.CLS_TO_INT[row["class"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:

            augmented = self.transforms(
                image=image, bboxes=target["boxes"], labels=labels
            )
            image = augmented["image"]
            target["boxes"] = torch.as_tensor(augmented["bboxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(augmented["labels"])

        return image, target, image_name

    def __len__(self):
        return len(self.images)
