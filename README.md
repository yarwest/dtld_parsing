# The DriveU Traffic Light Dataset (DTLD): Introduction and Comparison with Existing Datasets
This repository provides code for parsing the DriveU Traffic Light Dataset (DTLD), which is published in the course of our 2018 ICRA publication "The DriveU Traffic Light Dataset: Introduction and Comparison with Existing Datasets".

## Paper
Paper see https://ieeexplore.ieee.org/document/8460737.
## Download the dataset

The data can be downloaded from http://www.traffic-light-data.com/.

NEW v2 04/2021: json label format

    .
    ├── DTLD                 # DTLD
        ├── Berlin           # Contains all Routes of Berlin
        ├── Bochum           # Contains all routes of Bochum
        ├── Bremen           # Contains all routes of Bremen
        ├── Dortmund         # Contains all routes of Dortmund
        ├── Duesseldorf      # Contains all routes of Duesseldorf
        ├── Essen            # Contains all routes of Essen
        ├── Frankfurt        # Contains all routes of Frankfurt
        ├── Fulda            # Contains all routes of Fulda
        ├── Hannover         # Contains all routes of Hannover
        ├── Kassel           # Contains all routes of Kassel
        ├── Koeln            # Contains all routes of Cologne
        ├── DTLD_labels_v1.0 # Old labels (v1.0) in yml-format
        ├── DTLD_labels_v2.0 # New labels (v2.0) in json-format
        ├── LICENSE          # License
        └── README.md        # Readme

### Route structure
We separated each drive in one city into different routes

    .
    ├── Berlin                # Berlin
        ├── Berlin1           # First route
        ├── Berlin2           # Second route
        ├── Berlin3           # Third route
        ├── ...
### Sequence structure
We separated each route into several sequences. One sequence describes one unique intersection up to passing it. The foldername indicates date and time.

    .
    ├── Berlin 1                    # Route Berlin1
        ├── 2015-04-17_10-50-05     # First intersection
        ├── 2015-04-17_10-50-41     # Second intersection
        ├── ...

### Image structure
For each sequences, images and disparity images are available. Filename indicates time and date

    .
    ├── 2015-04-17_10-50-05                                      # Route Berlin1
        ├── DE_BBBR667_2015-04-17_10-50-13-633939_k0.tiff        # First left camera image
        ├── DE_BBBR667_2015-04-17_10-50-13-633939_nativeV2.tiff  # First disparity image
        ├── DE_BBBR667_2015-04-17_10-50-14-299876_k0.tiff        # Second left camera image
        ├── DE_BBBR667_2015-04-17_10-50-14-299876_nativeV2       # Second disparity image
        ├── ...
## Before starting
#### 1. Check our documentation
Documentation is stored at /dtld_parsing/doc/. We give insights into the data and explain how to interpret it.
#### 2. Change absolute paths
Do not forget to change the absolute paths of the images in all label files.

## Using the dataset

### Python

1. Download data & DTLD_Labels_v2.0.zip from https://cloudstore.uni-ulm.de/training/DTLD
2. Clone GitHub repository containing parsing scrips

`git clone https://github.com/julimueller/dtld_parsing`

3. Enter newly created directory

`cd dtld_parsing`

4. Create a virtual Python environment in which we can install dependencies required by the parsing scripts:

`python3 -m venv .venv`

5. After creation, activate the virtual environment like so:

`source .venv/bin/activate`

*You can verify that this was successful by running `which python3`, which shows the path of the python that you're currently using, if this shows a path ending in `/dtld_parsing/.venv/bin/python3` the virtual environment is being used successfully*

6. Run the setup script to install required dependencies using:

`python3 setup.py install`

7. Run the Python script to load the data, make sure that the label file & exported data are in the dtld_parsing folder, & provide the correct path to the label file `<LABEL_FILE_PATH>`:

`python3 python/load_dtld.py --label_file <LABEL_FILE_PATH> --calib_dir calibration/`

## Citation
Do not forget to cite our work for the case you used DTLD
### Citation:
```
@INPROCEEDINGS{8460737,
author={A. Fregin and J. Müller and U. Kreβel and K. Dietmayer},
booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
title={The DriveU Traffic Light Dataset: Introduction and Comparison with Existing Datasets},
year={2018},
volume={},
number={},
pages={3376-3383},
keywords={computer vision;image recognition;traffic engineering computing;DriveU traffic light dataset;traffic light recognition;autonomous driving;computer vision;University of Ulm Traffic Light Dataset;Daimler AG;Cameras;Urban areas;Benchmark testing;Lenses;Training;Visualization;Detectors},
doi={10.1109/ICRA.2018.8460737},
ISSN={2577-087X},
month={May},}

```
