import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os


def bc_loss(model, data, labels):
    action_preds = model(data)
    labels = labels.to(torch.long)
    loss = F.cross_entropy(action_preds, labels)

    return loss


@torch.no_grad()
def accuracy_fn(model, data, labels):
    action_preds = model(data)
    labels = labels.to(torch.long)
    _, predicted = torch.max(action_preds, 1)

    correct = (predicted == labels).sum().item()
    accuracy = float(correct) / len(labels)

    return accuracy


class EarlyStopping:
    """Early stopping utility to monitor validation metrics and stop training when improvement plateaus."""

    def __init__(self, patience=None, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, val_score):
        """Check if training should stop based on validation score."""
        if self.patience is None:
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            return True
        return False


def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn,
    accuracy_function, device='cpu', scheduler=None, early_stopping_patience=None,
    early_stopping_min_delta=0.0001, data_parallel=False):
    """Train a model using behavior cloning with optional early stopping and optional multi-GPU DataParallel."""
    torch.set_default_device(device)
    # Optional multi-GPU support via DataParallel
    if data_parallel and device.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    losses_train = []
    accuracy_validation = []
    early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(train_dataloader), desc="Epoch Progress", leave=False)

    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(train_dataloader, loss_fn, model, optimizer, epoch_progressbar, device)
        losses_train.append(avg_epoch_loss)

        avg_val_accuracy = validate_model(model, val_dataloader, accuracy_function, device)
        accuracy_validation.append(avg_val_accuracy)

        # Check early stopping
        if early_stopping(avg_val_accuracy):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {early_stopping.best_score:.4f}")
            break

        if scheduler is not None:
            scheduler.step(avg_epoch_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(train_loss=avg_epoch_loss, val_accuracy=avg_val_accuracy)

    progressbar.close()
    epoch_progressbar.close()

    return model, losses_train, accuracy_validation


def train_epoch(dataloader, loss_fn, model, optimizer, epoch_progressbar, device):
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
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = loss_fn(model, data, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

    return avg_epoch_loss


@torch.no_grad()
def validate_model(model, dataloader, accuracy_fn, device):
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

    for data, labels in dataloader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total_accuracy += accuracy_fn(model, data, labels)
        num_batches += 1

    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

    return avg_accuracy


if __name__ == '__main__':
    class SimpleMNISTModel(nn.Module):
        """Simple CNN model for MNIST classification."""

        def __init__(self, num_classes=10):
            super(SimpleMNISTModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def load_mnist_data(batch_size=64, data_dir='./data'):
        """Load MNIST dataset and return train/val dataloaders."""

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(data_dir, True, transform, download=True)
        test_dataset = torchvision.datasets.MNIST(data_dir, False, transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size, True, num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(test_dataset, batch_size, False, num_workers=2, pin_memory=True, persistent_workers=True)

        return train_loader, val_loader


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 512
    num_epochs = 25
    learning_rate = 0.001

    train_loader, val_loader = load_mnist_data(batch_size=batch_size)
    model = SimpleMNISTModel(num_classes=10)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Train model using behavior cloning
    print("\nStarting training...")
    trained_model, train_losses, val_accuracies = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        loss_fn=bc_loss,
        accuracy_function=accuracy_fn,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=3,  # Stop if no improvement for 3 epochs
        early_stopping_min_delta=0.001,  # Minimum improvement threshold
        data_parallel=(torch.cuda.is_available() and torch.cuda.device_count() > 1)
    )

    # Print results
    print("\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")
