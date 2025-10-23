import torch
from src.early_stopping import EarlyStopping


def train(model, optimizer, criterion, train_gen, val_gen, device):
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

    return (
        avg_train_losses,
        avg_val_losses,
        train_accuracies,
        val_accuracies,
        epoch + 1,
    )
