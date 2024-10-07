from torch.autograd import Variable
import torch

def train(trainData, model, criterion, optimizer, scheduler=None, trainLoss=[], device='cpu'):
    """
    Train the model for one epoch.
    
    Args:
        trainData (DataLoader): DataLoader for training batches.
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        trainLoss (list): List to record average loss per epoch.
        device (str): Device to use ('cpu', 'cuda', 'mps').
    """

    model.train()  # Set model to training mode
    epoch_loss = 0
    i = 0

    for batch_idx, (img, label) in enumerate(trainData):
        # Transfer data to the appropriate device
        img = Variable(img).to(device)
        label = Variable(label).to(device)

        # Forward pass
        out = model(img)
        loss = criterion(out, label)  # Compute loss
        epoch_loss += loss.item()

        # Backward pass and optimization step
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights

        # Update learning rate if a scheduler is provided
        if scheduler is not None:
            scheduler.step()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(trainData)}: Loss {loss.item():.4f}')

        i += 1

    # Average epoch loss
    avg_epoch_loss = epoch_loss / i
    print(f'Epoch Training Loss: {avg_epoch_loss:.4f}')

    # Save the average loss for the epoch
    if trainLoss is not None:
        trainLoss.append(avg_epoch_loss)

    # Print the learning rate if a scheduler is provided
    if scheduler is not None:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')


def validate(valData, model, criterion, buffer=None, valLoss=[], device='cpu'):
    """
    Validate the model.

    Args:
        valData (DataLoader): DataLoader for validation batches.
        model (torch.nn.Module): Trained model for validation.
        criterion (torch.nn.Module): Function to calculate loss.
        buffer (int, optional): Buffer added to the targeted grid when creating dataset.
        valLoss (list): List to record average loss for each epoch.
        device (str): Device to use ('cpu', 'cuda', 'mps').
    
    Returns:
        float: The average validation loss over the epoch.
    """

    model.eval()  # Set model to evaluation mode
    epoch_loss = 0
    i = 0

    with torch.no_grad():  # Disable gradients during validation
        for batch_idx, (img, label) in enumerate(valData):
            # Transfer data to the appropriate device
            img = img.to(device)
            label = label.to(device)

            # Forward pass
            out = model(img)

            # Apply buffer to mask the loss computation if buffer is provided
            if buffer:
                # Crop the center region to exclude the buffer
                out = out[:, :, buffer:-buffer, buffer:-buffer]
                label = label[:, buffer:-buffer, buffer:-buffer]

            # Compute loss
            loss = criterion(out, label)
            epoch_loss += loss.item()
            i += 1

            # Print validation progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(valData)}: Validation Loss {loss.item():.4f}')

    # Calculate the average validation loss
    avg_val_loss = epoch_loss / i
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # Store the average loss for this epoch
    if valLoss is not None:
        valLoss.append(avg_val_loss)

    return avg_val_loss
