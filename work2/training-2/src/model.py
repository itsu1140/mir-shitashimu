import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """
    すべての録音は同じ長さなので，入力バッチのサイズは `BxCxDxT` です．
    `B`: バッチサイズ (ここでは16に固定)
    `C`: 入力チャンネル数 (6種類の特徴量)
    `D`: 各タイムステップの次元数 (12個のMFCC・クロマ特徴およびそれらの∆と∆∆)
    `T`: フレーム数 (ウィンドウ処理後: 1290)
    """

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
            }
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


class ClassificationBlock(nn.Module):
    """
    非常に大きなCNNの出力を徐々に減らし，10 ユニットのソフトマックス層にします．
    また，ネットワークのこの段階でドロップアウトを適用します．
    サイズが小さくなるにつれて 3 つの連結層を用い，
    それぞれに ReLU 関数，最適化されるべきハイパーパラメータとして扱われるドロップアウトを持たせます．
    最後の連結層の出力はソフトマックス層を通過し，モデルの予測値を生成します．
    """

    def __init__(self, n_classes, n_linear, dropout):
        super().__init__()

        self.model = nn.ModuleDict(
            {
                "fc1": nn.Sequential(
                    nn.LazyLinear(n_linear[0]), nn.ReLU(), nn.Dropout(dropout[0])
                ),
                "fc2": nn.Sequential(
                    nn.LazyLinear(n_linear[1]), nn.ReLU(), nn.Dropout(dropout[1])
                ),
                "fc3": nn.Sequential(
                    nn.LazyLinear(n_linear[2]), nn.ReLU(), nn.Dropout(dropout[2])
                ),
                "softmax": nn.Sequential(
                    nn.LazyLinear(n_classes), nn.LogSoftmax(dim=1)
                ),
            }
        )

    def forward(self, x):
        # Fully-connected layers
        for i in range(1, 4):
            x = self.model[f"fc{i}"](x)

        # Log softmax class outputs
        return self.model["softmax"](x)


class GTZANCNN(nn.Module):
    """
    畳み込みブロックと分類ブロックを 1 つのモジュールにする．
    """

    def __init__(self, n_classes, n_channels, channel_widths, n_linear, dropout):
        super().__init__()
        self.conv = ConvolutionalBlock(n_channels, channel_widths)
        self.clf = ClassificationBlock(n_classes, n_linear, dropout)

    def forward(self, x):
        x = self.conv(x)
        return self.clf(x)
