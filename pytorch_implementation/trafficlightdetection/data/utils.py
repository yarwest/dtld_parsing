import os
import re
import numpy as np

import pandas as pd


def get_files_by_type(path, file_type, suffix=""):
    file_paths = []
    file_type = f".{file_type}"

    for current_dir, _, files in os.walk(path):
        for file_name in files:
            name, ext = os.path.splitext(file_name)
            if name.endswith(suffix) and ext == file_type:
                full_path = os.path.join(current_dir, file_name)
                file_paths.append(full_path)

    return file_paths


def transform_lisa_annotations(df):
    COLUMNS = ["filename", "class", "min_x", "min_y", "max_x", "max_y"]

    regexp = re.compile("(day|night)Sequence\d")

    def map_filename(val):
        root_dir_path, filename_path = os.path.split(val)
        clip_dir_path = filename_path.split("--")[0]

        # Ścieżka testowa (np. daySequence1)
        if regexp.search(filename_path):
            clip_dir_path = filename_path.split("--")[0]
            return f"{clip_dir_path}/{clip_dir_path}/frames/{filename_path}"

        # Ścieżka treningowa
        root_dir_path = root_dir_path.replace("Training", "Train")
        return f"{root_dir_path}/{root_dir_path}/{clip_dir_path}/frames/{filename_path}"

    df = df.iloc[:, :6]
    df.columns = COLUMNS
    df["filename"] = df["filename"].map(map_filename)
    return df


class LisaTrafficLightUtils:
    def __init__(self, root, annotations_type="BOX"):
        if annotations_type not in ["BOX", "BULB"]:
            raise ValueError("Annotation type must be BOX or BULB.")

        self.root = root
        self.annotations_type = annotations_type
        self.annotations_paths = self._init_annotations_paths()
        self.frames_paths = self._init_frames_paths()

    def get_annotations(self, time_of_day="both"):
        if time_of_day not in ["day", "night", "both"]:
            raise ValueError("Invalid time_of_day value")

        tods = [time_of_day] if time_of_day != "both" else ["day", "night"]

        annotations_dfs = []
        for tod in tods:
            for f in self.annotations_paths[tod]:
                df = pd.read_csv(f, sep=";")
                annotations_dfs.append(df)

        df = pd.concat(annotations_dfs)
        df = df.reset_index(drop=True)
        df = transform_lisa_annotations(df)
        return df

    def get_all_image_files(self, time_of_day="both"):
        if time_of_day not in ["day", "night", "both"]:
            raise ValueError("Invalid time_of_day value")

        if time_of_day == "both":
            return self.frames_paths["day"] + self.frames_paths["night"]
        else:
            return self.frames_paths[time_of_day]

    def _init_annotations_paths(self):
        annotations_root_path = os.path.join(self.root, "Annotations", "Annotations")

        annotations = {
            "day": [],
            "night": [],
        }

        paths = {
            "day": ["dayTrain", "daySequence1", "daySequence2"],
            "night": ["nightTrain", "nightSequence1", "nightSequence2"],
        }

        for item in annotations.items():
            time_of_day, files_list = item
            for path in paths[time_of_day]:
                joined_path = os.path.join(annotations_root_path, path)
                files = get_files_by_type(
                    joined_path,
                    "csv",
                    self.annotations_type,
                )
                files_list += files

        np_annotations = {
            "day": np.array(annotations["day"]),
            "night": np.array(annotations["night"]),
        }
        return np_annotations

    def _init_frames_paths(self):
        frames = {
            "day": [],
            "night": [],
        }

        paths = {
            "day": ["dayTrain", "daySequence1", "daySequence2"],
            "night": ["nightTrain", "nightSequence1", "nightSequence2"],
        }

        for item in frames.items():
            time_of_day, files_list = item
            for path in paths[time_of_day]:
                joined_path = os.path.join(self.root, path, path)
                files = get_files_by_type(
                    joined_path,
                    "jpg",
                )
                files_list += files

        np_frames = {
            "day": np.array(frames["day"]),
            "night": np.array(frames["night"]),
        }
        return np_frames
