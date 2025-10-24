# %%
import os
import shutil
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import torch
import wandb as wb
from optuna.integration.wandb import WeightsAndBiasesCallback
from torch import nn
from tqdm import tqdm

# %% [markdown]
"""
## 学習2
GTZANデータセットには，3秒の録音と30秒の録音の2つのバリエーションがあります．
30秒のデータセット情報は `features_30_sec.csv` に格納されており，ここからファイル名とラベルを取得できます．
"""

# %%
# データ読み込み (dataframe)
data_path = Path("GTZAN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(data_path / "features_30_sec.csv")

# ファイル名とラベルを取得
files, labels = df.filename, df.label

# %% [markdown]
"""
### トレーニング、バリデーション、テストセットの作成
データセットをトレーニング (80%)，バリデーション (10%)，テスト (10%) にランダムに分割します．

各分割のジャンルの分布が等しくなるようにします．
"""

# %%
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

# %% [markdown]
"""
### 特徴量のスケーリング
特徴量の正規化や標準化は，ネットワーク・パラメータを最適化する際の収束や速度の改善に役立ちます．
ここでは正規化を行います．

### 特徴量の結合
特徴量をCNNに渡す際，それぞれを個別のチャンネルとして積み重ね，時間軸上の任意のフレームが6チャンネル全てに渡る12の特徴に関する情報を含むようにします．
"""

# %%
D = 12  # Number of MFCCs and chroma features
T = 1290
# Number of frames
# after windowing a 30-second recording when generating MFCCs and chroma features
C = 6  # Number of channels - feature types (MFCC + ∆ + ∆∆, Chromagram + ∆ + ∆∆)


class GTZAN_setup(torch.utils.data.Dataset):
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
        file_path = data_path / "genres_original" / label / file

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

        # Only keep the first T frames
        # as there are some recordings that are slightly over 30s
        return (
            mfcc[:, :T],
            mfcc_d[:, :T],
            mfcc_dd[:, :T],
            chroma[:, :T],
            chroma_d[:, :T],
            chroma_dd[:, :T],
        )
        # Shape(s): [D x T]

    def scale(self, x):
        if self.scaler == "standardize":
            # Standardizing to zero mean and unit std. dev.
            return (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(
                dim=(1, 2),
                keepdim=True,
            )
        if self.scaler == "min-max":
            # Min-max scaling to [0, 1]
            return (x - x.amin(dim=(1, 2), keepdim=True)) / (
                x.amax(dim=(1, 2), keepdim=True) - x.amin(dim=(1, 2), keepdim=True)
            )
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
                x[5 - i],
                ax=ax,
                x_axis=("time" if i == 5 else None),
            )

        if path:
            fig.savefig(path, bbox_inches="tight")


# %%
# Create Dataset objects for each split
splits = {
    "train": GTZAN_setup(files_train, labels_train, n_features=D, scaler=None),
    "val": GTZAN_setup(files_val, labels_val, n_features=D, scaler=None),
    "test": GTZAN_setup(files_test, labels_test, n_features=D, scaler=None),
}

# %% [markdown]
"""
### 前処理
CNN の学習に便利なように，3つのデータセットに分割された各サンプルに対して前処理ステップを適用します．

結果が `.pt` ファイルに保存されるため，一度実行すれば再度の実行は不要です．
"""

# %%
split_dir_path = Path("./audio_split")
if split_dir_path.exists():
    shutil.rmtree(split_dir_path)
for split in tqdm(("train", "val", "test"), desc="Split"):
    # Create a sub-directory for each dataset split
    split_data_path = split_dir_path / split / "data"
    split_data_path.mkdir(parents=True, exist_ok=True)

    # Store the labels for the dataset split
    split_data = splits[split]
    labels[split].reset_index(drop=True).to_csv(split_dir_path / split / "labels.csv")

    # Preprocess each audio sample from the dataset split and store the resulting Tensor（about 20m）
    for i, data in tqdm(
        enumerate(split_data),
        total=len(split_data),
        desc="Sample",
        leave=False,
    ):
        torch.save(data, split_data_path / f"{i}.pt")

warnings.filterwarnings("ignore")

# Disable W&B logging
os.environ["WANDB_SILENT"] = "true"

# %% [markdown]
"""
### ラベルエンコーダの作成
`sklearn.preprocessing.LabelEncoder` を使い，10個のジャンル名を0〜9の整数にエンコードします．
"""

# %%
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

# %% [markdown]
"""
### 前処理された音声データの取得
このクラスでは，学習・検証・テストセット分割のため，前処理された特徴とそのラベルをフェッチします．
"""


# %%
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


# No. Features
n_features = 12

# Create Dataset objects for each split
split_data_path = Path("./audio_split")
train_set = GTZAN(split_data_path / "train", label_encoder)
val_set = GTZAN(split_data_path / "val", label_encoder)
test_set = GTZAN(split_data_path / "test", label_encoder)

# %% [markdown]
"""
### CNNモデルの作成
すべての録音は同じ長さなので，入力バッチのサイズは `BxCxDxT` です．

- `B`: バッチサイズ (ここでは16に固定)
- `C`: 入力チャンネル数 (6種類の特徴量)
- `D`: 各タイムステップの次元数 (12個のMFCC・クロマ特徴およびそれらの∆と∆∆)
- `T`: フレーム数 (ウィンドウ処理後: 1290)

#### 畳み込みブロック
4つの畳み込み層を使用します．

- 層のフィルタの高さを8，6，4，3とし，12個の入力を1つの特徴に縮小します．
- `T=1290` フレームという非常に大きな入力幅をよりコンパクトにまとめます．フィルター幅はハイパーパラメーターとして扱いますが，ストライドと水平パディングには固定値を使用します．
- 単純に特徴量を12個から1個に減らすだけでは情報量が減りすぎてしまうため，各フィルタの深さを大きくしていきます．フィルタの深さはハイパーパラメータとして扱います．

畳み込みの後，バッチ正規化を行い，正規化された出力をReLU活性化関数に通します．
最後の畳み込み層の出力は，`BxCLxWL` のサイズになります．
`CL` は最後のフィルターのチャンネル数、`WL` は最後の層からの出力の幅です．
これを分類ブロックに渡す前に，サイズ `Bx(WLxCL)` のベクトルのバッチに平坦化します．
"""


# %%
class ConvolutionalBlock(nn.Module):
    def __init__(self, n_channels, channel_widths):
        super().__init__()

        self.model = nn.ModuleDict(
            {
                "conv1": nn.Sequential(
                    nn.LazyConv2d(
                        n_channels[0],
                        kernel_size=(8, channel_widths[0]),
                        stride=(3, 4),
                        padding=(11, 4),
                    ),
                    nn.BatchNorm2d(n_channels[0]),
                    nn.ReLU(),
                ),
                "conv2": nn.Sequential(
                    nn.LazyConv2d(
                        n_channels[1],
                        kernel_size=(6, channel_widths[1]),
                        stride=(2, 3),
                        padding=(4, 3),
                    ),
                    nn.BatchNorm2d(n_channels[1]),
                    nn.ReLU(),
                ),
                "conv3": nn.Sequential(
                    nn.LazyConv2d(
                        n_channels[2],
                        kernel_size=(4, channel_widths[2]),
                        stride=(1, 2),
                        padding=(0, 2),
                    ),
                    nn.BatchNorm2d(n_channels[2]),
                    nn.ReLU(),
                ),
                "conv4": nn.Sequential(
                    nn.LazyConv2d(
                        n_channels[3],
                        kernel_size=(3, channel_widths[3]),
                        stride=(1, 1),
                        padding=(0, 1),
                    ),
                    nn.BatchNorm2d(n_channels[3]),
                    nn.ReLU(),
                ),
                "flatten": nn.Flatten(),
            },
        )

        self.show_shapes = False

    def forward(self, x):
        # Convolutional layers
        for i in range(1, 5):
            x = self.model[f"conv{i}"](x)
            if self.show_shapes:
                print(f"conv{i}: {x.shape}")

        # Flattened output
        return self.model["flatten"](x)


# %% [markdown]
"""
### 分類ブロック
非常に大きなCNNの出力を徐々に減らし，10ユニットのソフトマックス層にします．

また，ネットワークのこの段階でドロップアウトを適用します．

サイズが小さくなるにつれて3つの連結層を用い，それぞれにReLU活性化関数と，最適化されるべきハイパーパラメータとして扱われるドロップアウト確率を持たせます．

最後の連結層の出力はソフトマックス層を通過し，モデルの予測値を生成します．
"""


# %%
class ClassificationBlock(nn.Module):
    def __init__(self, n_classes, n_linear, dropout):
        super().__init__()

        self.model = nn.ModuleDict(
            {
                "fc1": nn.Sequential(
                    nn.LazyLinear(n_linear[0]),
                    nn.ReLU(),
                    nn.Dropout(dropout[0]),
                ),
                "fc2": nn.Sequential(
                    nn.LazyLinear(n_linear[1]),
                    nn.ReLU(),
                    nn.Dropout(dropout[1]),
                ),
                "fc3": nn.Sequential(
                    nn.LazyLinear(n_linear[2]),
                    nn.ReLU(),
                    nn.Dropout(dropout[2]),
                ),
                "softmax": nn.Sequential(
                    nn.LazyLinear(n_classes),
                    nn.LogSoftmax(dim=1),
                ),
            },
        )

    def forward(self, x):
        # Fully-connected layers
        for i in range(1, 4):
            x = self.model[f"fc{i}"](x)

        # Log softmax class outputs
        return self.model["softmax"](x)


# %% [markdown]
"""
### 畳み込みブロック・分類ブロックをまとめる
この2つのブロックを1つのモジュールにまとめます．
"""


# %%
class GTZANCNN(nn.Module):
    def __init__(self, n_classes, n_channels, channel_widths, n_linear, dropout):
        super().__init__()

        self.model = nn.ModuleDict(
            {
                "conv_block": ConvolutionalBlock(n_channels, channel_widths),
                "clf_block": ClassificationBlock(n_classes, n_linear, dropout),
            },
        )

    def forward(self, x):
        x = self.model["conv_block"](x)
        return self.model["clf_block"](x)


# %% [markdown]
"""
### 学習とハイパーパラメータの最適化
学習と検証のループを定義します．
結果を比較するために，各エポックのバッチにおける平均的な学習と検証の損失および精度を記録します．

ここで `EarlyStopping` コールバックにより，10エポックの間検証損失が改善されない場合は学習ループを停止します．
"""


# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        trace_func (function): trace print function.
                        Default: print

        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(
            #     f"EarlyStopping counter: {self.counter} out of {self.patience}"
            # )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...",
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# %%
def train_2(model, optimizer, criterion, train_gen, val_gen):
    n_epochs = 100

    # Average training/validation losses over batches per epoch
    avg_train_losses, avg_val_losses = [], []

    # Training/validation accuracy over epochs
    train_accuracies, val_accuracies = [], []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=False)

    for epoch in range(n_epochs):
        # Training loop
        model.train()
        train_correct, train_losses = [], []
        for batch, labels in train_gen:
            batch, labels = batch.to(device), labels.to(device)
            # Reset the optimizer
            optimizer.zero_grad()
            # Calculate predictions for batch
            log_prob = model(batch)
            y_pred = torch.argmax(log_prob, dim=1)
            # Calculate and back-propagate loss
            train_loss = criterion(log_prob, labels)
            train_losses.append(train_loss.item())
            train_loss.backward()
            # Store label comparisons for accuracy
            train_correct.append(labels == y_pred)
            # Update the optimizer
            optimizer.step()

        # Store average training loss and accuracy for epoch
        avg_train_losses.append(torch.Tensor(train_losses).mean().item())
        train_accuracies.append(torch.cat(train_correct).float().mean().item())

        # Validation loop
        model.eval()
        val_correct, val_losses = [], []
        with torch.no_grad():
            for batch, labels in val_gen:
                batch, labels = batch.to(device), labels.to(device)
                # Calculate predictions for batch
                log_prob = model(batch)
                y_pred = torch.argmax(log_prob, dim=1)
                # Calculate loss
                val_loss = criterion(log_prob, labels)
                val_losses.append(val_loss.item())
                # Store label comparisons for accuracy
                val_correct.append(labels == y_pred)

        # Store average validation loss and accuracy for epoch
        avg_val_losses.append(torch.Tensor(val_losses).mean().item())
        val_accuracies.append(torch.cat(val_correct).float().mean().item())

        # Update the early stopping object with average validation loss
        early_stopping(avg_val_losses[epoch], model)
        if early_stopping.early_stop:
            break

    return avg_train_losses, avg_val_losses, train_accuracies, val_accuracies, epoch + 1


# %% [markdown]
"""
### ハイパーパラメータ最適化のための目的関数を定義
"""


# %%
def objective(trial):
    # Suggest output channel depths
    n_channels = (
        trial.suggest_categorical("conv1_depth", (32, 64)),
        trial.suggest_categorical("conv2_depth", (64, 128)),
        trial.suggest_categorical("conv3_depth", (128, 256)),
        trial.suggest_categorical("conv4_depth", (256, 512)),
    )

    # Suggest channel widths
    channel_widths = (
        trial.suggest_categorical("conv1_width", (16, 32)),
        trial.suggest_categorical("conv2_width", (8, 16)),
        trial.suggest_categorical("conv3_width", (4, 8)),
        trial.suggest_categorical("conv4_width", (2, 4)),
    )

    # Suggest fully-connected units
    n_linear = (
        trial.suggest_categorical("fc1", (256, 512)),
        trial.suggest_categorical("fc2", (128, 256)),
        trial.suggest_categorical("fc3", (64, 128)),
    )

    # Suggest dropout probability
    dropout = (
        trial.suggest_uniform("p1", 0.1, 0.8),
        trial.suggest_uniform("p2", 0.1, 0.8),
        trial.suggest_uniform("p3", 0.1, 0.8),
    )

    # Suggest a learning rate and weight decay for the optimizer
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-3)

    # Initialize the model and send it to GPU
    model = GTZANCNN(n_classes, n_channels, channel_widths, n_linear, dropout).to(
        device,
    )
    trial.set_user_attr(key="model", value=model)

    # Set Adam optimizer and negative log-likelihood loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.NLLLoss()

    # Create training, validation and test set batch iterators
    train_gen = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )
    val_gen = torch.utils.data.DataLoader(
        val_set,
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )

    # Train model and evaluate validation NLL
    avg_train_losses, avg_val_losses, train_accuracies, val_accuracies, epochs = (
        train_2(
            model,
            optimizer,
            criterion,
            train_gen,
            val_gen,
        )
    )

    # Record training/validation losses and accuracies for this trial
    trial.set_user_attr(key="train_losses", value=avg_train_losses)
    trial.set_user_attr(key="val_losses", value=avg_val_losses)
    trial.set_user_attr(key="train_accuracies", value=train_accuracies)
    trial.set_user_attr(key="val_accuracies", value=val_accuracies)
    trial.set_user_attr(key="epochs", value=epochs)

    # Return the average validation loss over the batches of the last epoch
    return avg_val_losses[-1]


# %% [markdown]
"""
### Optunaの初期化とstudyの実行

70分ほどかかります．
PCがスリープに入らないよう注意してください．
"""

# %%
# Create a new Optuna study for hyper-parameter optimization
seed = 0
sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction="minimize", sampler=sampler)

# W&B integration - Initializes a new job for keeping track of hyper-parameter optimization
wb_callback = WeightsAndBiasesCallback(
    metric_name="val_loss",
    wandb_kwargs={"project": "gtzan-cnn", "name": "final-search"},
)


# Callback to save the model that had the best Optuna trial
def model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


# Run the hyper-parameter search
study.optimize(
    objective,
    n_trials=25,
    show_progress_bar=True,
    callbacks=[wb_callback, model_callback],
)
wb.finish()

# %% [markdown]
"""
最適なトライアルの構成を見ることができ，それに対応するモデルのトーチサマリーを表示できます．
"""

# %%
# Fetch the best trial and model
trial = study.best_trial
model = study.user_attrs["best_model"]

# %% [markdown]
"""
# 評価
最良のモデルにおける学習セットと検証セットの損失と精度のグラフです．
"""

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Show training/validation loss curves
sns.lineplot(
    data=pd.DataFrame(
        {
            "Training": trial.user_attrs["train_losses"],
            "Validation": trial.user_attrs["val_losses"],
        },
    ),
    ax=axs[0],
)
axs[0].set(title="Loss (Negative Log-Likelihood)", xlabel="Epoch", ylabel="Loss")
axs[0].xaxis.get_major_locator().set_params(integer=True)

# Show training/validation accuracy curves
axs[1] = sns.lineplot(
    data=pd.DataFrame(
        {
            "Training": trial.user_attrs["train_accuracies"],
            "Validation": trial.user_attrs["val_accuracies"],
        },
    ),
)
axs[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
axs[1].xaxis.get_major_locator().set_params(integer=True)

plt.show()

# Create the summary run for writing loss and accuracy curves (for the best model) to W&B
with wb.init(project="gtzan-cnn", name="final-summary", job_type="logging") as summary:
    for i in range(trial.user_attrs["epochs"]):
        summary.log(
            {
                "final_train_loss": trial.user_attrs["train_losses"][i],
                "final_val_loss": trial.user_attrs["val_losses"][i],
                "final_train_accuracy": trial.user_attrs["train_accuracies"][i],
                "final_val_accuracy": trial.user_attrs["val_accuracies"][i],
            },
            step=i,
        )

# %% [markdown]
"""
テストセットで最適モデルの性能を評価し，テスト精度と混同行列を求めます．
"""

# %%
# Toggle evaluation mode
model.eval()

# Create test set batch iterator
test_gen = torch.utils.data.DataLoader(
    test_set,
    batch_size=len(test_set),
    num_workers=2,
)

# Retrieve test set as a single batch and send to GPU
batch, labels = next(iter(test_gen))
batch, labels = batch.to(device), labels.to(device)

# Calculate predictions for test set
y = model(batch)
y_pred = torch.argmax(y, dim=1)

# Calculate accuracy
torch.mean((labels == y_pred).float())

# Plot the test confusion matrix with per-class precision and recall values

fig, ax = plt.subplots(figsize=(11, 11))
cm = sklearn.metrics.confusion_matrix(labels.cpu().numpy(), y_pred.cpu().numpy())
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# 各クラスの適合率 (precision) と再現率 (recall) を表示
report = sklearn.metrics.classification_report(
    labels.cpu().numpy(),
    y_pred.cpu().numpy(),
    target_names=classes,
)
# print(report)

plt.show()
