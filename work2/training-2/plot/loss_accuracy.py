"""
Plot the test confusion matrix with per-class precision and recall values
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb as wb


def plot_loss_accuracy(trial):
    """
    最良のモデルにおける学習セットと検証セットの損失と精度のグラフです．
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Show training/validation loss curves
    sns.lineplot(
        data=pd.DataFrame(
            {
                "Training": trial.user_attrs["train_losses"],
                "Validation": trial.user_attrs["val_losses"],
            }
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
            }
        )
    )
    axs[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axs[1].xaxis.get_major_locator().set_params(integer=True)

    plt.savefig("loss_accuracy.png")

    # Create the summary run for writing loss and accuracy curves (for the best model) to W&B
    with wb.init(
        project="gtzan-cnn", name="final-summary", job_type="logging"
    ) as summary:
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
