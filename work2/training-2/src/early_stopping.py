"""
学習と検証のループを定義します．

結果を比較するために，各エポックのバッチにおける平均的な学習と検証の損失および精度を記録します.
ここで `EarlyStopping` コールバックにより，
10エポックの間検証損失が改善されない場合は学習ループを停止します．
"""

from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        path: Path | str = "checkpoint.pt",
        trace_func: callable = print,
    ):
        """
        Difine constructor

        Args:
        patience: How long to wait after last time validation loss improved.
                        Default: 7
        verbose: If True, prints a message for each validation loss improvement.
                        Default: False
        delta: Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        path (pathlike): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        trace_func: trace print function.
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
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model.",
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
