import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def to_device_tensor(data, labels):
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long, device=data.device)
    else:
        labels = labels.to(device=data.device, dtype=torch.long)

    return labels


def bc_loss(model, data, labels):
    action_preds = model(data)
    labels = to_device_tensor(data, labels)
    loss = F.cross_entropy(action_preds, labels)

    return loss


def accuracy_fn(model, data, labels):
    action_preds = model(data)
    labels = to_device_tensor(data, labels)
    _, predicted = torch.max(action_preds, 1)

    correct = (predicted == labels).sum()
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
    - accuracies_validation (list): List of average validation accuracies per epoch.
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
        progressbar.set_postfix(train_loss=avg_epoch_loss, val_accuracy=avg_val_accuracy.item())

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
        loss = loss_fn(model, data, labels)

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

        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    # Example usage with MNIST dataset
    print("Behavior Cloning Example with MNIST")
    print("=" * 40)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001

    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader = load_mnist_data(batch_size=batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = SimpleMNISTModel(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
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
        scheduler=scheduler
    )

    # Print results
    print("\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")

    # Test final model
    print("\nEvaluating on test set...")
    final_accuracy = validate_model(trained_model, val_loader, accuracy_fn)
    print(f"Test accuracy: {final_accuracy:.4f}")

    # Example prediction
    print("\nExample prediction:")
    trained_model.eval()
    with torch.no_grad():
        data_iter = iter(val_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images[:5])  # First 5 samples
        _, predicted = torch.max(outputs, 1)

        print("True labels:", labels[:5].cpu().numpy())
        print("Predictions:", predicted.cpu().numpy())
