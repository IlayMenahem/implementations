import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional, Any

from ray import tune
from ray.tune.schedulers import ASHAScheduler


def loss_and_accuracy(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    action_preds = model(data)
    labels = labels.to(torch.long)
    loss = F.cross_entropy(action_preds, labels)
    _, predicted = torch.max(action_preds, 1)

    correct = (predicted == labels).sum().item()
    accuracy = float(correct) / len(labels)

    return loss, accuracy


class EarlyStopping:
    def __init__(self, patience: Optional[int] = None, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False

    def __call__(self, val_score: float) -> bool:
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


def train_epoch(dataloader: DataLoader, loss_and_accuracy_fn: Callable, model: nn.Module,
    optimizer: optim.Optimizer, epoch_progressbar: tqdm, device: str) -> tuple[float, float]:
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for data, labels in dataloader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, accuracy = loss_and_accuracy_fn(model, data, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
        epoch_progressbar.update(1)
        epoch_progressbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_epoch_accuracy = float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0

    return avg_epoch_loss, avg_epoch_accuracy


@torch.no_grad()
def validate_model(model: nn.Module, dataloader: DataLoader, loss_and_accuracy_fn: Callable,
    device: str) -> tuple[float, float]:
    model.eval()
    total_accuracy = 0.0
    total_loss = 0.0
    num_batches = 0

    for data, labels in dataloader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss, accuracy = loss_and_accuracy_fn(model, data, labels)

        total_accuracy += accuracy
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches else 0.0
    return avg_loss, avg_accuracy


def train_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
    num_epochs: int, optimizer: optim.Optimizer, loss_and_accuracy_fn: Callable,
    device: str = 'cpu', scheduler: Optional[Any] = None, early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0001, data_parallel: bool = False,
    epoch_reporter: Optional[Callable[[int, float, float, float, float], None]] = None
) -> tuple[nn.Module, list, list, list, list]:
    """
    Train loop with optional:
      - Early stopping
      - LR scheduler
      - DataParallel
      - Epoch-wise reporting callback (for Ray Tune or custom logging)

    Parameters
    ----------
    epoch_reporter : callable | None
        If provided, called every epoch as:
          epoch_reporter(epoch_index, train_loss, val_loss,
                         train_accuracy, val_accuracy)
        You can use this to forward metrics to Ray Tune via tune.report().
    """
    torch.set_default_device(device)
    if data_parallel and device.startswith('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    losses_train = []
    accuracy_train = []
    losses_validation = []
    accuracy_validation = []

    early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

    progressbar = tqdm(total=num_epochs, desc="Training", unit="epoch")
    epoch_progressbar = tqdm(total=len(train_dataloader), desc="Epoch Progress", leave=False)

    for epoch in range(num_epochs):
        avg_epoch_loss, avg_epoch_accuracy = train_epoch(train_dataloader, loss_and_accuracy_fn, model, optimizer, epoch_progressbar, device)
        avg_val_loss, avg_val_accuracy = validate_model(model, val_dataloader, loss_and_accuracy_fn, device)

        losses_train.append(avg_epoch_loss)
        accuracy_train.append(avg_epoch_accuracy)
        losses_validation.append(avg_val_loss)
        accuracy_validation.append(avg_val_accuracy)

        if epoch_reporter is not None:
            epoch_reporter(epoch, avg_epoch_loss, avg_val_loss, avg_epoch_accuracy, avg_val_accuracy)

        if early_stopping(avg_val_accuracy):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {early_stopping.best_score:.4f}")
            break

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        epoch_progressbar.reset()
        progressbar.update(1)
        progressbar.set_postfix(train_loss=f"{avg_epoch_loss:.4f}", val_acc=f"{avg_val_accuracy:.4f}")

    progressbar.close()
    epoch_progressbar.close()

    return model, losses_train, accuracy_train, losses_validation, accuracy_validation


def tune_hyperparameters(config: dict[str, Any], gpus_per_trial: float = 0.0):
    """
    Perform hyperparameter tuning using Ray Tune with ASHA scheduler.

    Required keys in config:
      - model_class : nn.Module class (not instance)
      - model_args  : dict (optional) kwargs for model_class
      - get_loaders : function(batch_size) -> (train_loader, val_loader)
      - loss_and_accuracy : callable(model, data, labels) -> (loss, accuracy)
      - max_num_epochs : int
      - num_trials : int (number of Ray samples)

    Search-space parameters can be Ray Tune search objects (e.g., tune.choice, tune.loguniform)
    or fixed values:
      - batch_size
      - learning_rate
      - early_stopping_patience
      - early_stopping_min_delta
      - any other meta-hparams you use inside _train_model.

    Metrics reported per epoch:
      - loss (validation loss)          -> optimization target (mode=min)
      - accuracy (validation accuracy)
      - train_loss
      - train_accuracy
      - epoch
      - lr (current learning rate of first param group)

    Notes:
      - ASHA uses the per-report "training_iteration" counter, which increments
        automatically each time tune.report(...) is called.
    """
    def _train_model(parameters: dict[str, Any]):
        model_class = parameters["model_class"]

        model_args = {}
        for key, value in parameters.items():
            if key.startswith("model_args/"):
                arg_name = key.split("/", 1)[1]
                model_args[arg_name] = value

        model = model_class(**model_args)

        batch_size = parameters.get("batch_size", 64)
        learning_rate = parameters.get("learning_rate", 0.001)
        num_epochs = parameters.get("max_num_epochs", 10)
        early_stopping_patience = parameters.get("early_stopping_patience", None)
        early_stopping_min_delta = parameters.get("early_stopping_min_delta", 0.0001)

        get_loaders = parameters["get_loaders"]
        num_workers = parameters.get("num_workers", 2)
        train_loader, val_loader = get_loaders(batch_size, num_workers)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_and_accuracy_fn = parameters["loss_and_accuracy"]

        device = 'cuda' if (torch.cuda.is_available() and gpus_per_trial > 0) else 'cpu'

        def reporter(epoch_idx: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
            current_lr = optimizer.param_groups[0]["lr"]
            tune.report({
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "epoch": epoch_idx + 1,
                    "lr": current_lr
                })

        # Train
        train_model(model, train_loader, val_loader, num_epochs, optimizer, loss_and_accuracy_fn,
            device, scheduler=None, early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            data_parallel=(device.startswith('cuda') and torch.cuda.device_count() > 1),
            epoch_reporter=reporter
        )

    # Scheduler
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2
    )

    # Build tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(_train_model),
            resources={"cpu": config.get("cpus_per_trial", 2), "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=config["num_trials"],
        ),
        param_space=config,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="loss", mode="min")

    print("\n===== Best Trial =====")
    print(f"Config: {best_result.config}")
    print(f"Best Validation Loss: {best_result.metrics['loss']:.6f}")
    print(f"Best Validation Accuracy: {best_result.metrics['accuracy']:.4f}")
    return best_result


if __name__ == '__main__':
    class SimpleMNISTModel(nn.Module):
        """Simple CNN model for MNIST classification."""

        def __init__(self, num_classes: int = 10, conv1_channels: int = 32, conv2_channels: int = 64, fc_hidden: int = 128, dropout_p: float = 0.5):
            super(SimpleMNISTModel, self).__init__()
            self.conv1 = nn.Conv2d(1, conv1_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(conv2_channels * 7 * 7, fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, num_classes)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
            x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    def load_mnist_data(batch_size: int, num_workers: int = 2, data_dir: str = './data'):
        """Load MNIST dataset and return train/val dataloaders."""

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(data_dir, True, transform, download=True)
        test_dataset = torchvision.datasets.MNIST(data_dir, False, transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size, True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(test_dataset, batch_size, False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

        return train_loader, val_loader


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_batch_size = 512
    base_epochs = 5
    base_learning_rate = 0.001

    train_loader, val_loader = load_mnist_data(base_batch_size)
    model = SimpleMNISTModel(num_classes=10)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=base_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print("\n==== Starting baseline training (no Ray Tune) ====")
    trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, base_epochs, optimizer,
        loss_and_accuracy, device, scheduler,
        early_stopping_patience=3, early_stopping_min_delta=0.001,
        data_parallel=(torch.cuda.is_available() and torch.cuda.device_count() > 1)
    )

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accuracies, 'b-', label='Train Acc')
    ax2.plot(epochs_range, val_accuracies, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")

    tuning_config = {
        # Required structural keys
        "model_class": SimpleMNISTModel,
        "model_args": {
            "num_classes": 10,
            "conv1_channels": tune.choice([16, 32, 48]),
            "conv2_channels": tune.choice([32, 64, 96]),
            "fc_hidden": tune.choice([64, 128, 256]),
            "dropout_p": tune.choice([0.3, 0.5])
        },
        "get_loaders": load_mnist_data,
        "loss_and_accuracy": loss_and_accuracy,

        # Optimization hyperparameters
        "batch_size": tune.choice([64, 128, 256]),
        "learning_rate": tune.loguniform(5e-5, 5e-3),

        # Training control
        "max_num_epochs": 5,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0005,

        # Ray sampling
        "num_trials": 2,

        # Resource hint
        "cpus_per_trial": 1,
        "num_workers": 2
    }

    best = tune_hyperparameters(tuning_config, gpus_per_trial=0.0)

    print("\nBest trial summary:")
    print(best.config)
    print(f"Loss: {best.metrics['loss']:.6f}, Accuracy: {best.metrics['accuracy']:.4f}")
