import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MultiLabelStratifiedSplitter:
    def __init__(
        self,
        collection,
        index_col,
        target_col,
        train_size=0.7,
        valid_size=0.15,
        test_size=0.15,
        random_state=None,
    ):
        self.collection = collection
        self.index_col = index_col
        self.target_col = target_col
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.random_state = random_state

        self._train_values = None
        self._valid_values = None
        self._test_values = None

    def split(self):
        train_values = []
        valid_values = []
        test_values = []

        # TODO: fix?
        collection_df = pd.DataFrame(self.collection)
        df = pd.DataFrame(pd.json_normalize(self.collection, ['labels']))
        df = df.reset_index(drop=True).join(collection_df).reset_index(drop=True)

        sorted_targets = df[f'attributes.{self.target_col}'].value_counts(ascending=True)
        adjusted_train_size = self.train_size / (1.0 - self.test_size)

        for target in sorted_targets.index.tolist():
            target_rows = df.loc[df[f'attributes.{self.target_col}'] == target]
            target_values = target_rows[self.index_col].unique()

            target_train_valid_indices, target_test_indices = train_test_split(
                np.arange(len(target_values)),
                test_size=self.test_size,
                random_state=self.random_state,
            )

            target_train_indices, target_valid_indices = train_test_split(
                target_train_valid_indices,
                train_size=adjusted_train_size,
                random_state=self.random_state,
            )

            train_values += target_values[target_train_indices].tolist()
            valid_values += target_values[target_valid_indices].tolist()
            test_values += target_values[target_test_indices].tolist()

            df = df.loc[~df[self.index_col].isin(target_values)]

        unique_index_values = pd.Series(collection_df[self.index_col].unique())

        self._train_values = unique_index_values.loc[
            unique_index_values.isin(train_values)
        ]
        self._valid_values = unique_index_values.loc[
            unique_index_values.isin(valid_values)
        ]
        self._test_values = unique_index_values.loc[
            unique_index_values.isin(test_values)
        ]

    def train_values(self):
        return self._train_values

    def valid_values(self):
        return self._valid_values

    def test_values(self):
        return self._test_values

    def train_indices(self):
        return self._train_values.index.tolist()

    def valid_indices(self):
        return self._valid_values.index.tolist()

    def test_indices(self):
        return self._test_values.index.tolist()
