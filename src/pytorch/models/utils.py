import time
import mlflow
import torch
from configs import DEVICE
from tqdm.auto import tqdm

def get_experiment_id(name):
    """
    Get the experiment ID given an experiment name. 
    Create a new experiment if it doesn't exist.

    Args:
        name (str): Name of the experiment.

    Returns:
        str: The ID of the experiment.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id

def evaluate_per_task(model, criterion, support_images, support_labels, query_images, query_labels):
    """
    Evaluate model performance per task using a given criterion.

    Args:
        model: The PyTorch model.
        criterion: The PyTorch loss function.
        support_images (torch.Tensor): The support images.
        support_labels (torch.Tensor): The support labels.
        query_images (torch.Tensor): The query images.
        query_labels (torch.Tensor): The query labels.

    Returns:
        loss (torch.Tensor): The loss value.
        correct (int): The number of correct predictions.
        total (int): The total number of predictions.
    """
    # Forward pass
    classification_scores = model(support_images, support_labels, query_images)
    # Calculate loss
    loss = criterion(classification_scores, query_labels)

    correct = (torch.max(classification_scores.detach().data, 1)[1]==query_labels).sum().item()
    total = query_labels.shape[0]
    return loss, correct, total

def train_step(model, criterion, optimizer, epoch, train_loader):
    """
    Perform a training step.

    Args:
        model: The PyTorch model.
        criterion: The PyTorch loss function.
        optimizer: The PyTorch optimizer.
        epoch (int): The current epoch.
        train_loader: The PyTorch dataloader for training data.

    Returns:
        train_loss (float): The average training loss.
        train_acc (float): The training accuracy.
    """
    train_loss, train_correct, train_total = 0, 0, 0

    # Put model in train mode
    model.train()
    
    # Loop through data loader data batches
    tqdm_train = tqdm(
        enumerate(train_loader),
        desc=f"Training Epoch {epoch}",
        total=len(train_loader)
    )

    for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
        support_images, support_labels, query_images, query_labels = (
            support_images.to(DEVICE), support_labels.to(DEVICE),
            query_images.to(DEVICE), query_labels.to(DEVICE)
        )
        optimizer.zero_grad()
        loss, correct, total = evaluate_per_task(
            model, criterion,
            support_images, support_labels,
            query_images, query_labels
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += correct
        train_total+= total

        tqdm_train.set_postfix(
            {
                'train_loss': train_loss / (episode_index + 1),
                'train_acc' : train_correct / train_total
            }
        )

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    return train_loss, train_acc

def val_step(model, criterion, epoch, val_loader):
    """
    Perform a validation step.

    Args:
        model: The PyTorch model.
        criterion: The PyTorch loss function.
        epoch (int): The current epoch.
        val_loader: The PyTorch dataloader for validation data.

    Returns:
        val_loss (float): The average validation loss.
        val_acc (float): The validation accuracy.
    """
    val_loss, val_correct, val_total = 0, 0, 0

    # Put model in eval mode
    model.eval()
    
    # Loop through data loader data batches
    tqdm_val = tqdm(
        enumerate(val_loader),
        desc=f"Validation Epoch {epoch}",
        total=len(val_loader)
    )

    # Turn on inference context manager
    with torch.no_grad():
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_val:
            support_images, support_labels, query_images, query_labels = (
                support_images.to(DEVICE), support_labels.to(DEVICE),
                query_images.to(DEVICE), query_labels.to(DEVICE)
            )
            loss, correct, total = evaluate_per_task(
                model, criterion,
                support_images, support_labels,
                query_images, query_labels
            )

            val_loss += loss.item()
            val_correct += correct
            val_total+= total

            tqdm_val.set_postfix(
                {
                    'val_loss': val_loss / (episode_index + 1),
                    'val_acc' : val_correct / val_total
                }
            )

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    return val_loss, val_acc

def train_per_epoch(model, criterion, optimizer, epoch, train_loader, val_loader):
    """
    Perform a complete training and validation step for one epoch.

    Args:
        model: The PyTorch model.
        criterion: The PyTorch loss function.
        optimizer: The PyTorch optimizer.
        epoch (int): The current epoch.
        train_loader: The PyTorch dataloader for training data.
        val_loader: The PyTorch dataloader for validation data.

    Returns:
        train_loss (float): The average training loss.
        val_loss (float): The average validation loss.
        train_acc (float): The training accuracy.
        val_acc (float): The validation accuracy.
        train_epoch_time (float): The time taken for the training step.
        val_epoch_time (float): The time taken for the validation step.
    """
    # Perform training step and time it
    start = time.time()
    train_loss, train_acc = train_step(model, criterion, optimizer, epoch, train_loader)
    end = time.time()
    train_epoch_time = end - start

    # Perform validation step and time it
    start = time.time()
    val_loss, val_acc = val_step(model, criterion, epoch, val_loader)
    end = time.time()
    val_epoch_time = end - start

    return (
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        train_epoch_time,
        val_epoch_time
    )