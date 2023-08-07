import argparse
import os
import logging
import numpy as np
import sys

import pytorch_lightning as pl
import yaml

from trafficlightdetection.neuralnet.model import FasterRcnnModel
from trafficlightdetection.data.datamodule import LTLDDataModule

np.set_printoptions(suppress=True)


# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_dataloader_num_workers(configuration):
    use_cpu_count = configuration["dataloaders"]["worker_per_cpu"]
    if use_cpu_count:
        return os.cpu_count()
    else:
        return configuration["dataloaders"]["number_of_workers"]


def is_tuning_required(configuration):
    return configuration["trainer"]["tune"] is True


def train_and_test(configuration):
    seed = pl.seed_everything(configuration["seed"], workers=True)

    model = FasterRcnnModel(
        learning_rate=configuration["model"]["learning_rate"],
        iou_threshold=configuration["model"]["iou_threshold"],
        num_classes=configuration["model"]["num_of_classes"],
        trainable_backbone_layers=configuration["model"]["trainable_backbone_layers"],
        early_stopping_patience=configuration["model"]["early_stopping_patience"],
        early_stopping_min_delta=configuration["model"]["early_stopping_min_delta"],
    )

    datamodule = LTLDDataModule(
        train_label_path=configuration["dataset"]["train_label_file"],
        test_label_path=configuration["dataset"]["test_label_file"],
        batch_size=configuration["dataloaders"]["batch_size"],
        num_workers=get_dataloader_num_workers(configuration),
        random_state=seed,
    )

    trainer = pl.Trainer(
        default_root_dir=configuration["trainer"]["root_dir"],
        max_epochs=configuration["trainer"]["max_epochs"],
        limit_train_batches=configuration["trainer"]["limit_train_set"],
        limit_val_batches=configuration["trainer"]["limit_valid_set"],
        limit_test_batches=configuration["trainer"]["limit_test_set"],
        fast_dev_run=configuration["trainer"]["fast_dev_run"],
        deterministic=configuration["trainer"]["deterministic_trainer"],
        devices=2,
        #TODO? maybe select the appropriate Accelerator by recognizing the machine you are on: accelerator="auto"
        accelerator="auto",
        log_every_n_steps=1,
    )

    if is_tuning_required(configuration):
        trainer.tune(model, datamodule)

    print("1. Training model")
    trainer.fit(model, datamodule)

    print("2. Testing model")
    trainer.test(datamodule=datamodule, ckpt_path="best")


def main(args):
    with open(args.configuration_path, "r") as file:
        configuration = yaml.safe_load(file)

    if args.print:
        print(configuration)



    train_and_test(configuration)


def parse_args():
    """
    Parse command line arguments

    :return: Collection of parsed arguments
    """

    parser = argparse.ArgumentParser(
        prog="train.py",
        description="""
        This program handles the training of a Faster R-CNN
        model for the task of traffic light detection.
        """,
    )

    parser.add_argument(
        "--configuration_path",
        type=str,
        default="configuration.yml",
        required=False,
        help="A path to the YAML file with training configuration",
    )

    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print parsed configuration before training.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
