import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch


def plot_heatmap(
    model: torch.nn.Module,
    test_set: torch.utils.data.Dataset,
    device: torch.device,
    classes: list[str],
):
    """
    テストセットで最適モデルの性能を評価し，テスト精度と混同行列を求めます．
    """
    # Toggle evaluation mode
    model.eval()

    # Create test set batch iterator
    test_gen = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), num_workers=2
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
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # 各クラスの適合率 (precision) と再現率 (recall) を表示
    report = sklearn.metrics.classification_report(
        labels.cpu().numpy(), y_pred.cpu().numpy(), target_names=classes
    )
    # print(report)

    plt.savefig("confusion_matrix.png")
