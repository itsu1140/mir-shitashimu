from pathlib import Path

import pandas as pd
import torch


class GTZAN(torch.utils.data.Dataset):
    """Fetches data from the preprocessed GTZAN dataset."""

    def __init__(self, split, encoder):
        data_path = self.path(split, "data")
        self.data_files = [data_path / item.name for item in data_path.iterdir()]
        self.data_files.sort(key=self.get_id)
        self.labels = pd.read_csv(self.path(split, "labels.csv"), index_col=0).squeeze(
            "columns",
        )
        self.encoder = encoder

    def get_id(self, file_path):
        return int(file_path.stem)  # remove extension

    def path(self, *sub):
        return Path().joinpath(*sub)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file = self.data_files[index]

        # Fetch the preprocessed input
        x = torch.load(file)

        # Fetch the audio label and encode it
        label = self.labels.iloc[index]
        y = self.encoder.transform([label]).item()

        return x, y
