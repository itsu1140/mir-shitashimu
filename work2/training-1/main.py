import torch
from src.make_dataset import get_dataset
from src.model import Music_net
from src.plot import plot_loss_accuracy
from torch import nn
from tqdm import tqdm


def train(model, device, train_loader, validation_loader, epochs):
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
                data = data.to(device)
                target = target.to(device)
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
                data = data.to(device)
                target = target.to(device)
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


def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this code to perform best, "
            "if possible, changing the machine you are connected to.",
        )
    else:
        print("GPU is enabled")

    return device


def main():
    device = get_device()
    net = Music_net().to(device)
    train_loader, val_loader = get_dataset()
    train_loss, train_acc, val_loss, val_acc = train(
        model=net,
        device=device,
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=50,
    )
    plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc)


if __name__ == "__main__":
    main()
