from pathlib import Path

import pandas as pd
import sklearn
import torch
from GTZAN_setup import GTZAN_setup
from tqdm import tqdm


def random_split(files, labels):
    """
    データセットをトレーニング (80%)，バリデーション (10%)，テスト (10%) にランダムに分割します．
    各分割のジャンルの分布が等しくなるようにします．
    """

    seed = 0
    files_train, files_val_test, labels_train, labels_val_test = (
        sklearn.model_selection.train_test_split(
            files,
            labels,
            stratify=labels,
            train_size=0.8,
            random_state=seed,  # train : (val + test) = 8 : 2
        )
    )

    files_val, files_test, labels_val, labels_test = (
        sklearn.model_selection.train_test_split(
            files_val_test,
            labels_val_test,
            stratify=labels_val_test,
            test_size=0.5,  # val : test = 1 : 1
            random_state=seed,
        )
    )

    labels = {"train": labels_train, "val": labels_val, "test": labels_test}

    # Create Dataset objects for each split
    splits = {
        "train": GTZAN_setup(
            files_train, labels_train, n_features=GTZAN_setup.D, scaler=None
        ),
        "val": GTZAN_setup(
            files_val, labels_val, n_features=GTZAN_setup.D, scaler=None
        ),
        "test": GTZAN_setup(
            files_test, labels_test, n_features=GTZAN_setup.D, scaler=None
        ),
    }
    return labels, splits


def preprocess(labels: dict, splits: dict):
    """
    CNN の学習に便利なように，3つのデータセットに分割された各サンプルに対して前処理ステップを適用します．
    結果が `.pt` ファイルに保存されるため，一度実行すれば再度の実行は不要です．
    """
    split_dir_path = Path("./audio_split")
    for split in tqdm(("train", "val", "test"), desc="Split"):
        # Create a sub-directory for each dataset split
        split_data_path = split_dir_path / split / "data"
        split_data_path.mkdir(parents=True, exist_ok=True)

        # Store the labels for the dataset split
        split_data = splits[split]
        labels[split].reset_index(drop=True).to_csv(
            split_dir_path / split / "labels.csv"
        )

        # Preprocess each audio sample from the dataset split and store the resulting Tensor
        # about 5m
        for i, data in tqdm(
            enumerate(split_data),
            total=len(split_data),
            desc="Sample",
            leave=False,
        ):
            torch.save(data, split_data_path / f"{i}.pt")


def main():
    """
    GTZAN Dataset には，3秒の録音と30秒の録音の2つのバリエーションがあります．
    30秒のデータセット情報は `features_30_sec.csv` に格納されており，ここからファイル名とラベルを取得できます．
    """

    data_path = Path("GTZAN")
    if not data_path.exists():
        data_path.symlink_to("/work2/itsuki/shitashimu/GTZAN", target_is_directory=True)

    # データ読み込み (dataframe)
    df = pd.read_csv(data_path / "features_30_sec.csv")

    # ファイル名とラベルを取得
    files, labels = df.filename, df.label

    labels, splits = random_split(files, labels)
    preprocess(labels, splits)


if __name__ == "__main__":
    main()
