import os

import cv2
import pandas as pd
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np
import json
import logging
import os
import sys

from trafficlightdetection.data.utils import LisaTrafficLightUtils

class LTLDDataset(torch.utils.data.Dataset):
    # Im mniejszy numer, tym bardziej liczna klasa
    STATE_TO_INT = {
        "green": 1,
        "red": 2,
        "yellow": 3,
        "red_yellow": 4,
        "off": 5,
        "unknown": 6,
    }

    INT_TO_STATE = {v: k for k, v in STATE_TO_INT.items()}

    def __init__(self, file_path, transforms):
        self.file_path = file_path
        self.transforms = transforms
        self.images = []

        if os.path.exists(self.file_path) is not None:
            label_file_extension = os.path.splitext(self.file_path)[1]
            if label_file_extension == ".json":
                logging.info("Opening DriveuDatabase from file: {}"
                            .format(self.file_path))
                with open(self.file_path, "r") as fp:
                    images = json.load(fp)
            elif label_file_extension == ".yml":
                logging.exception("Yaml support is deprecated. Either use the new .json label files (from download URL received after registration) or checkout <git checkout v1.0> to parse yaml")
                sys.exit(1)
                return False
            else:
                logging.exception("Label file with extension {} not supported. Please use json!".format(label_file_extension))
                sys.exit(1)
                return False
        else:
            logging.exception(
                "Opening DriveuDatabase from File: {} "
                "failed. File or Path incorrect.".format(self.file_path)
            )
            sys.exit(1)
            return False
        
        image_dfs = []
        # for image in images["images"]:
        #     image_dfs.append(pd.DataFrame.from_dict(image))
        self.images = images["images"]#pd.concat(image_dfs)

    def __getitem__(self, idx):
        image_data = self.images[idx]
        image_path = image_data["image_path"]
        image_name = os.path.basename(image_path)

        # Load image from file path, do debayering and shift
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if os.path.splitext(image_path)[1] == ".tiff":
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2BGR)
                # Images are saved in 12 bit raw -> shift 4 bits
                image = np.right_shift(image, 4)
                image = image.astype(np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float32")
            image /= 255

        else:
            logging.error("Image {} not found. Please check image file paths!".format(image_path))
            sys.exit(1)
            return False, np.array()
        
        image_annotations = pd.DataFrame(image_data["labels"])

        boxes = []
        labels = []
        for _, row in image_annotations.iterrows():
            attributes = row["attributes"]
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])
            boxes.append([x, y, x+w, y+h])
            labels.append(self.STATE_TO_INT[attributes["state"]])

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
