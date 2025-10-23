import matplotlib.pyplot as plt


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
    fig.savefig("loss_accuracy_plot.png")
