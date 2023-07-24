
import argparse
import logging
import sys
import os
import json

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

__author__ = "Yarno Boelens"


np.set_printoptions(suppress=True)


# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", help="DTLD label files (.json)", type=str, required=True)
    return parser.parse_args()

def print_separator(character="=", length=60):
    print(character * length)

STATES = [ "red", "yellow", "red_yellow", "green", "off", "unknown" ]
DIRECTIONS = [ "front", "back", "left", "right" ]

def main(args):
    logging.info("Trying to load data")
    print("Loading...")

    # Load dataset
    if os.path.exists(args.label_file) is not None:
        logging.info("Opening DriveuDatabase from file: {}"
                    .format(args.label_file))
        with open(args.label_file, "r") as fp:
            images = json.load(fp)

    else:
        logging.exception(
            "Opening DriveuDatabase from File: {} "
            "failed. File or Path incorrect.".format(args.label_file)
        )
        sys.exit(1)
        return False
    

    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('./out/')

    files_count = 0

    img_dfs = []
    label_dfs = []

    for image in images["images"]:

        img_dfs.append(pd.DataFrame(image))

        files_count += 1

        label_dfs.append(pd.DataFrame.from_dict(image["labels"]))

    print("File count: ", files_count)

    print_unique_lights(pd.concat(label_dfs))
    print_separator()
    print_state_distribution(pd.concat(label_dfs))
    print_separator()
    print_direction_distribution(pd.concat(label_dfs))

def print_missing_values():
    # .isnull().sum()
    print("")

def print_unique_lights(df):
    lights = df["track_id"].nunique()
    light_count = df["track_id"].value_counts()
    print("Light count: ", lights)
    print(light_count)

def print_state_distribution(df):
    attributes_df = pd.DataFrame([at for at in df.loc[:,"attributes"] ])

    states = attributes_df["state"].unique()
    print("Traffic light states:")
    print(states)

    states_count = attributes_df["state"].value_counts()
    states_freq = attributes_df["state"].value_counts(normalize=True)
    states_params = pd.DataFrame({"Count": states_count, "Frequency": states_freq})
    print("Probability distribution and class size:")
    print(states_params)

    states_count.plot(
        kind="bar",
        title=f'Numbers of individual states of traffic lights in the dataset',
        rot=45,
    )
    plt.savefig('./out/traffic_states_distribution.png')

def print_direction_distribution(df):
    attributes_df = pd.DataFrame([at for at in df.loc[:,"attributes"] ])

    directions = attributes_df["direction"].unique()
    print("Traffic light directions:")
    print(directions)

    directions_count = attributes_df["direction"].value_counts()
    directions_freq = attributes_df["direction"].value_counts(normalize=True)
    directions_params = pd.DataFrame({"Count": directions_count, "Frequency": directions_freq})
    print("Probability distribution and class size:")
    print(directions_params)

    directions_count.plot(
        kind="bar",
        title=f'Numbers of individual directions of traffic lights in the dataset',
        rot=45,
    )
    plt.savefig('./out/traffic_directions_distribution.png')

if __name__ == "__main__":
    main(parse_args())
