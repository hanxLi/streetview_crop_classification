from torch.autograd import Variable
from loss import AleatoricLoss, BalancedCrossEntropyUncertaintyLoss
import torch

def train(trainData, model, criterion, optimizer, scheduler=None, trainLoss=None, 
          device='cpu', use_ancillary=False):
    """
    Train the model for one epoch.

    Args:
        trainData (DataLoader): DataLoader for training batches.
        model (torch.nn.Module): The model to train.
        criterion (nn.Module or function): The loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        trainLoss (list, optional): List to record average loss per epoch.
        device (str): Device to use ('cpu', 'cuda', 'mps').
        use_ancillary (bool): Whether to use ancillary data.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()  # Set model to training mode
    epoch_loss = 0
    i = 0

    for _, batch in enumerate(trainData):
        if use_ancillary:
            img, ancillary_data, label = batch
            ancillary_data = ancillary_data.to(device)
        else:
            img, label = batch  # No ancillary data

        # Transfer data to the appropriate device
        img = img.to(device)
        label = label.to(device)

        # Forward pass
        if use_ancillary:
            output = model(img, ancillary_data)  # Pass both image and ancillary data
        else:
            output = model(img)  # Pass only image

        # Check if the model outputs uncertainty (tuple with logits and log_var)
        if isinstance(output, tuple):
            logits, log_var = output
            loss = criterion(logits, label, log_var)  # Use aleatoric loss
        else:
            logits = output  # For models without uncertainty
            loss = criterion(logits, label)  # Use standard loss

        epoch_loss += loss.item()

        # Backward pass and optimization step
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights

        i += 1

    # Average epoch loss
    avg_epoch_loss = epoch_loss / i
    print(f'Epoch Training Loss: {avg_epoch_loss:.4f}')

    # Save the average loss for the epoch if trainLoss list is provided
    if trainLoss is not None:
        trainLoss.append(avg_epoch_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current Learning Rate: {current_lr:.6f}')

    return avg_epoch_loss


def validate(valData, model, criterion, valLoss=None, device='cpu', use_ancillary=False):
    """
    Validate the model for one epoch.

    Args:
        valData (DataLoader): DataLoader for validation batches.
        model (torch.nn.Module): Trained model for validation.
        criterion (nn.Module or function): Function to calculate loss.
        valLoss (list, optional): List to record average loss for each epoch.
        device (str): Device to use ('cpu', 'cuda', 'mps').
        use_ancillary (bool): Whether to use ancillary data.

    Returns:
        float: The average validation loss over the epoch.
    """
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0
    i = 0

    with torch.no_grad():  # Disable gradients during validation
        for _, batch in enumerate(valData):
            if use_ancillary:
                img, ancillary_data, label = batch
                ancillary_data = ancillary_data.to(device)
            else:
                img, label = batch  # No ancillary data

            # Transfer data to the appropriate device
            img = img.to(device)
            label = label.to(device)

            # Forward pass
            if use_ancillary:
                output = model(img, ancillary_data)  # Pass both image and ancillary data
            else:
                output = model(img)  # Pass only image

            # Check if the model outputs uncertainty (tuple with logits and log_var)
            if isinstance(output, tuple):
                logits, log_var = output
                loss = criterion(logits, label, log_var)  # Use aleatoric loss
            else:
                logits = output  # For models without uncertainty
                loss = criterion(logits, label)  # Use standard loss

            epoch_loss += loss.item()
            i += 1

    # Calculate the average validation loss
    avg_val_loss = epoch_loss / i
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Store the average loss for this epoch if valLoss list is provided
    if valLoss is not None:
        valLoss.append(avg_val_loss)

    return avg_val_loss