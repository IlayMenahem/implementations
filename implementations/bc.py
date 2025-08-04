import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def bc_loss(model, labels, data):
    action_preds = model(data)
    action_preds = pad_sequence(action_preds, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    loss = F.cross_entropy(action_preds, labels)

    return loss


def accuracy_fn(model, labels, data):
    action_preds = model(data)
    action_preds = pad_sequence(action_preds, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)

    _, predicted = torch.max(action_preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)

    return accuracy


def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn,
    accuracy_function, device='cpu', scheduler=None):
    '''
    Train a model using behavior cloning.

    Args:
    - model (torch.nn.Module): The model model to train.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - val_dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - accuracy_function (callable): Function to compute accuracy of the model.
    - device (str): Device to run the training on ('cpu', 'cuda', or 'mps').
    - scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.

    Returns:
    - model (torch.nn.Module): The trained model model.
    - losses_train (list): List of average training losses per epoch.
    - losses_validation (list): List of average validation losses per epoch.
    '''
    torch.set_default_device(device)
    model.to(device)

    losses_train = []
    accuracy_validation = []

    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(train_dataloader), desc="Epoch Progress", leave=False)

    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(train_dataloader, loss_fn, model, optimizer, epoch_progressbar)
        losses_train.append(avg_epoch_loss)

        # Validate after each epoch
        avg_val_accuracy = validate_model(model, val_dataloader, accuracy_function)
        accuracy_validation.append(avg_val_accuracy)

        if scheduler is not None:
            scheduler.step(avg_epoch_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(train_loss=avg_epoch_loss, val_accuracy=avg_val_accuracy)

    return model, losses_train, accuracy_validation


def train_epoch(dataloader, loss_fn, model, optimizer, epoch_progressbar):
    '''
    Train the model for one epoch.

    Args:
    - dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
    - loss_fn (callable): Loss function to compute the loss of a batch.
    - model (torch.nn.Module): The model model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model.
    - epoch_progressbar (tqdm): Progress bar for the current epoch.

    Returns:
    - avg_epoch_loss (float): Average loss over the epoch.
    '''
    model.train()
    epoch_losses = []

    for data, labels in dataloader:
        loss = loss_fn(model, labels, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

    return avg_epoch_loss


def validate_model(model, dataloader, accuracy_fn):
    '''
    Validate the model on a validation dataset.

    Args:
    - model (torch.nn.Module): The model to validate.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing validation data.
    - accuracy_fn (callable): Function to compute accuracy of the model.

    Returns:
    - avg_accuracy (float): Average accuracy over the validation dataset.
    '''
    model.eval()
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, labels in dataloader:
            total_accuracy += accuracy_fn(model, labels, data)
            num_batches += 1

    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else float('inf')

    return avg_accuracy.item()
