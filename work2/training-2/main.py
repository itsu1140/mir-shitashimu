import os
import warnings
from pathlib import Path

import sklearn
import torch
from plot.heatmap import plot_heatmap
from plot.loss_accuracy import plot_loss_accuracy
from src.make_dataset import GTZAN
from src.optuna_study_trial import optuna_study


def main():
    # Hide warnings from experimental 'Lazy' PyTorch modules
    warnings.filterwarnings("ignore")

    # Disable W&B logging
    os.environ["WANDB_SILENT"] = "true"

    # `sklearn.preprocessing.LabelEncoder` を使い，10個のジャンル名を0〜9の整数にエンコードします．
    # All of the possible genres
    classes = (
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    )
    n_classes = len(classes)

    # Fit the encoder for the genre labels
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(classes)

    # No. Features
    n_features = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create Dataset objects for each split
    split_data_path = Path("./audio_split")
    train_set = GTZAN(split_data_path / "train", label_encoder)
    val_set = GTZAN(split_data_path / "val", label_encoder)
    test_set = GTZAN(split_data_path / "test", label_encoder)

    trial, model = optuna_study(device, train_set, val_set, n_classes)
    plot_loss_accuracy(trial)
    plot_heatmap(model, test_set, device, classes)


if __name__ == "__main__":
    main()
