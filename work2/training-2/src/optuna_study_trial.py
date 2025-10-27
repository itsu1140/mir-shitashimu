import optuna
import torch
import wandb as wb
from optuna.integration.wandb import WeightsAndBiasesCallback
from src.model import GTZANCNN
from src.train import train
from torch import nn


# Callback to save the model that had the best Optuna trial
def model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["model"])


def optuna_study(
    device: torch.device,
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    n_classes: int,
):
    ### ハイパーパラメータ最適化のための目的関数を定義
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
            train(model, optimizer, criterion, train_gen, val_gen, device)
        )

        # Record training/validation losses and accuracies for this trial
        trial.set_user_attr(key="train_losses", value=avg_train_losses)
        trial.set_user_attr(key="val_losses", value=avg_val_losses)
        trial.set_user_attr(key="train_accuracies", value=train_accuracies)
        trial.set_user_attr(key="val_accuracies", value=val_accuracies)
        trial.set_user_attr(key="epochs", value=epochs)

        # Return the average validation loss over the batches of the last epoch
        return avg_val_losses[-1]

    ### Optunaの初期化とstudyの実行
    # 70分ほどかかります．PCがスリープに入らないよう注意してください．

    # Create a new Optuna study for hyper-parameter optimization
    seed = 0
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # W&B integration - Initializes a new job for keeping track of hyper-parameter optimization
    wb_callback = WeightsAndBiasesCallback(
        metric_name="val_loss",
        wandb_kwargs={"project": "gtzan-cnn", "name": "final-search"},
    )

    # Run the hyper-parameter search
    study.optimize(
        objective,
        n_trials=25,
        show_progress_bar=True,
        callbacks=[wb_callback, model_callback],
    )
    wb.finish()

    # Fetch the best trial and model
    # 最適なトライアルの構成を見ることができ，それに対応するモデルのトーチサマリーを表示できます．
    trial = study.best_trial
    model = study.user_attrs["best_model"]
    return trial, model
