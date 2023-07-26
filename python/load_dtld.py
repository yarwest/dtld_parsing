from __future__ import print_function

import argparse
import logging
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from dtld_parsing.driveu_dataset import DriveuDatabase


__author__ = "Andreas Fregin, Julian Mueller and Klaus Dietmayer"
__maintainer__ = "Julian Mueller"
__email__ = "julian.mu.mueller@daimler.com"


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
    parser.add_argument(
        "--data_base_dir",
        default="",
        help="only use this if the image file paths in the"
        "label files are not up to date. Do NOT change the"
        "internal DTLD folder structure!",
        type=str,
    )
    return parser.parse_args()


def main(args):

    # Load database
    database = DriveuDatabase(args.label_file)
    if not database.open(args.data_base_dir):
        return False

    # create axes
    ax1 = plt.subplot(111)
    plt.axis('off')

    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('./out/')

    fig = plt.gcf()
    fig.set_size_inches(2048/72, 1024/72)

    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)

    # Visualize image by image
    for idx_d, img in enumerate(database.images):

        # Get color image without labels
        img_color = img.get_labeled_image()

        img_color = img_color[..., ::-1]
        if idx_d == 0:
            im1 = ax1.imshow(img_color)
        im1.set_data(img_color)
        plt.pause(0.001)
        
        fileName = img.file_path.replace(".tiff","").replace(".","").replace("/", "_")[1:]

        plt.savefig(f'./out/{fileName}.png', bbox_inches='tight', pad_inches = 0, dpi=72)
        print("Stored file: ", fileName)


if __name__ == "__main__":
    main(parse_args())
