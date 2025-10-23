import torch
from data_directory import get_data_directories
from torchvision import datasets, transforms


def get_dataset():
    data_directories = get_data_directories()
    train_dir, val_dir = data_directories[0], data_directories[2]
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
    return train_loader, val_loader
