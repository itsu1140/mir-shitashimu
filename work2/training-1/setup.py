import random
import shutil
from pathlib import Path

from src.make_dataset import get_data_directories


def main():
    train_data_path = Path("image_split")
    data_path = Path("GTZAN")
    if train_data_path.exists():
        shutil.rmtree(train_data_path)
    Path.mkdir(train_data_path)

    # 学習用, テスト用, 検証用のディレクトリ作成
    spectrograms_dir = data_path / "images_original"
    train_data_directories = get_data_directories()

    # 全ジャンル分
    for directory in train_data_directories:
        if directory.exists():
            shutil.rmtree(directory)
        Path.mkdir(directory)

    for genre in spectrograms_dir.iterdir():
        # 学習, テスト, 検証データに分割
        src_file_paths = list(genre.glob("*.png"))
        random.shuffle(src_file_paths)
        test_files = src_file_paths[0:10]
        val_files = src_file_paths[10:20]
        train_files = src_file_paths[20:]
        shuffled_directories = [train_files, test_files, val_files]

        # 画像の保存先フォルダを生成
        for directory in train_data_directories:
            Path.mkdir(directory / genre.name)

        # 学習, テスト用画像のハードリンクの作成
        for data_dir, files in zip(
            train_data_directories,
            shuffled_directories,
            strict=False,
        ):
            for f in files:
                (data_dir / genre.name / f.name).hardlink_to(f)


if __name__ == "__main__":
    main()
