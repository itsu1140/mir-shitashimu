from pathlib import Path

import torch
import librosa
import matplotlib.pyplot as plt

DATA_PATH = Path("GTZAN")


class GTZAN_setup(torch.utils.data.Dataset):
    """
    特徴量の正規化や標準化は，ネットワーク・パラメータを最適化する際の収束や速度の改善に役立ちます．
    ここでは正規化を行います．

    特徴量をCNNに渡す際，それぞれを個別のチャンネルとして積み重ね，
    時間軸上の任意のフレームが6チャンネル全てに渡る12の特徴に関する情報を含むようにします．
    """

    D = 12  # Number of MFCCs and chroma features
    # Number of frames
    # (after windowing a 30-second recording when generating MFCCs and chroma features)
    T = 1290
    # Number of channels - feature types (MFCC + ∆ + ∆∆, Chromagram + ∆ + ∆∆)
    C = 6

    def __init__(self, files, labels, n_features, scaler=None, **args):
        super().__init__()
        self.files = files
        self.labels = labels
        self.n_features = n_features
        self.scaler = scaler
        self.args = args

    def fetch(self, index):
        # Fetch the file path and corresponding label
        file, label = self.files.iloc[index], self.labels.iloc[index]
        file_path = DATA_PATH / "genres_original" / label / file

        # Load the audio and encode the label
        x, _ = librosa.load(file_path, **self.args)

        return x

    def transform(self, x):
        # Generate D MFCCs & ∆ + ∆∆
        mfcc = librosa.feature.mfcc(y=x, n_mfcc=(self.n_features + 1), **self.args)[1:]
        mfcc_d, mfcc_dd = (
            librosa.feature.delta(mfcc),
            librosa.feature.delta(mfcc, order=2),
        )
        # Shape(s): D x T

        # Generate chroma features & ∆ + ∆∆
        chroma = librosa.feature.chroma_stft(y=x, **self.args)
        chroma_d, chroma_dd = (
            librosa.feature.delta(chroma),
            librosa.feature.delta(chroma, order=2),
        )
        # Shape(s): D x T

        # Only keep the first T frames (as there are some recordings that are slightly over 30s)
        return (
            mfcc[:, : GTZAN_setup.T],
            mfcc_d[:, : GTZAN_setup.T],
            mfcc_dd[:, : GTZAN_setup.T],
            chroma[:, : GTZAN_setup.T],
            chroma_d[:, : GTZAN_setup.T],
            chroma_dd[:, : GTZAN_setup.T],
        )
        # Shape(s): [D x T]

    def scale(self, x):
        if self.scaler == "standardize":
            # Standardizing to zero mean and unit std. dev.
            return (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(
                dim=(1, 2), keepdim=True
            )
        elif self.scaler == "min-max":
            # Min-max scaling to [0, 1]
            return (x - x.amin(dim=(1, 2), keepdim=True)) / (
                x.amax(dim=(1, 2), keepdim=True) - x.amin(dim=(1, 2), keepdim=True)
            )
        else:
            # No scaling
            return x

    def to_tensor(self, x):
        # Scale these features and combine them into a multi-channel input
        return torch.Tensor(x)
        # Shape: C x D x T

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = self.fetch(index)
        x = self.transform(x)
        x = self.to_tensor(x)
        x = self.scale(x)
        return x

    def plot(self, index, figsize=None, path=None):
        x = self.fetch(index)
        x = self.transform(x)

        fig, axs = plt.subplots(3, 2, figsize=figsize)

        # MFCC, ∆ and ∆∆
        axs[0][0].set_title("MFCC")
        librosa.display.specshow(x[0], ax=axs[0][0])
        axs[1][0].set_title("Δ")
        librosa.display.specshow(x[1], ax=axs[1][0])
        axs[2][0].set_title("ΔΔ")
        librosa.display.specshow(x[2], ax=axs[2][0], x_axis="time")

        # Chroma, ∆ and ∆∆
        axs[0][1].set_title("Chroma")
        librosa.display.specshow(x[3], ax=axs[0][1], y_axis="chroma")
        axs[1][1].set_title("Δ")
        librosa.display.specshow(x[4], ax=axs[1][1], y_axis="chroma")
        axs[2][1].set_title("ΔΔ")
        librosa.display.specshow(x[5], ax=axs[2][1], x_axis="time", y_axis="chroma")

        plt.tight_layout()

        if path:
            fig.savefig(path, bbox_inches="tight")

    def plot_stacked(self, index, x_step=0.03, y_step=-0.06, figsize=None, path=None):
        x = self.fetch(index)
        x = self.transform(x)

        fig = plt.figure(figsize=figsize)

        for i in range(6):
            ax = fig.add_axes([i * x_step, i * y_step, 1.0, 1.0])
            librosa.display.specshow(
                x[5 - i], ax=ax, x_axis=("time" if i == 5 else None)
            )

        if path:
            fig.savefig(path, bbox_inches="tight")
