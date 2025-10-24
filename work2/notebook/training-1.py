# %%
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

# %% [markdown]
"""
## 学習1
音声ファイルのメルスペクトログラムからジャンル分類を行う
"""


# %%
data_path = Path("GTZAN")


def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this notebook to perform best, "
            "if possible, changing the machine you are connected to.",
        )
    else:
        print("GPU is enabled in this notebook.")

    return device


device = get_device()


# lossとaccuracyをプロット
def plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc):
    epochs = len(train_loss)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list(range(epochs)), train_loss, label="Training Loss")
    ax1.plot(list(range(epochs)), validation_loss, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Epoch vs Loss")
    ax1.legend()

    ax2.plot(list(range(epochs)), train_acc, label="Training Accuracy")
    ax2.plot(list(range(epochs)), validation_acc, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Epoch vs Accuracy")
    ax2.legend()
    fig.set_size_inches(15.5, 5.5)
    plt.show()


# %%
# 学習用，テスト用，検証用のディレクトリ作成
spectrograms_dir = data_path / "images_original"
train_data_path = Path("image_split")
if not train_data_path.exists():
    Path.mkdir(train_data_path)
train_data_directories = [
    train_data_path / "train",
    train_data_path / "test",
    train_data_path / "val",
]
train_dir = train_data_directories[0]
test_dir = train_data_directories[1]
val_dir = train_data_directories[2]

# 全ジャンル分
for directory in train_data_directories:
    if directory.exists():
        shutil.rmtree(directory)
    Path.mkdir(directory)

for genre in spectrograms_dir.iterdir():
    # 学習・テスト・検証データに分割
    src_file_paths = []
    for image in genre.glob("*.png"):
        src_file_paths.append(image)
    random.shuffle(src_file_paths)
    test_files = src_file_paths[0:10]
    val_files = src_file_paths[10:20]
    train_files = src_file_paths[20:]
    shuffled_directories = [train_files, test_files, val_files]

    # 画像の保存先フォルダを生成
    for directory in train_data_directories:
        Path.mkdir(directory / genre.name)

    # 学習・テスト用画像のハードリンクの作成
    for train_dir, files in zip(
        train_data_directories,
        shuffled_directories,
        strict=False,
    ):
        for f in files:
            (train_dir / genre.name / f.name).hardlink_to(f)

# %%
# データ読み込み
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ],
    ),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=25,
    shuffle=True,
    num_workers=0,
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transforms.Compose(
        [
            transforms.ToTensor(),
        ],
    ),
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=25,
    shuffle=True,
    num_workers=0,
)

# %%
# CNN


class music_net(nn.Module):
    def __init__(self):
        """Intitalize neural net layers"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.fc1 = nn.Linear(in_features=9856, out_features=10)

        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.batchnorm4 = nn.BatchNorm2d(num_features=64)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)

        self.dropout = nn.Dropout(p=0.3, inplace=False)

    def forward(self, x):
        # Conv layer 1.
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv layer 2.
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv layer 3.
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv layer 4.
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv layer 5.
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Fully connected layer 1.
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.softmax(x, dim=1)

        return x


# %%
def train_1(model, device, train_loader, validation_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    with tqdm(range(epochs), unit="epoch") as tepochs:
        tepochs.set_description("Training")
        for _ in tepochs:
            model.train()
            # keep track of the running loss
            running_loss = 0.0
            correct, total = 0, 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                # Get the model output (call the model with the data from this batch)
                output = model(data)
                # Zero the gradients out)
                optimizer.zero_grad()
                # Get the Loss
                loss = criterion(output, target)
                # Calculate the gradients
                loss.backward()
                # Update the weights (using the training step of the optimizer)
                optimizer.step()

                tepochs.set_postfix(loss=loss.item())
                running_loss += loss  # add the loss for this batch

                # get accuracy
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            # append the loss for this epoch
            train_loss.append(running_loss.detach().cpu().item() / len(train_loader))
            train_acc.append(correct / total)

            # evaluate on validation data
            model.eval()
            running_loss = 0.0
            correct, total = 0, 0

            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                tepochs.set_postfix(loss=loss.item())
                running_loss += loss.item()
                # get accuracy
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            validation_loss.append(running_loss / len(validation_loader))
            validation_acc.append(correct / total)

    return train_loss, train_acc, validation_loss, validation_acc


# %%
# 学習実行 (5分くらい)
net = music_net().to(device)
train_loss, train_acc, validation_loss, validation_acc = train_1(
    net,
    device,
    train_loader,
    val_loader,
    50,
)

# Detach tensors from GPU
plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc)
