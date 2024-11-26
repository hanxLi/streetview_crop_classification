import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss with inverse square root of class frequency weighting.
    
    Params:
        class_weights (torch.Tensor or None): Predefined class weights for each class.
        ignore_index (int): Class index to ignore.
        reduction (str): Reduction method to apply ('mean', 'sum', or 'none').
    '''

    def __init__(self, class_weights=None, ignore_index=-100, reduction='mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights  # Predefined class-wise weights (if any)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predict, target):
        # Get unique classes and their counts in the target
        unique, unique_counts = torch.unique(target, return_counts=True)

        # Remove the ignored index from calculations
        valid_mask = unique != self.ignore_index
        unique = unique[valid_mask]
        unique_counts = unique_counts[valid_mask]

        # Calculate dynamic class weights using inverse square root of frequency
        ratio = unique_counts.float() / torch.numel(target)  # Frequency ratio
        dynamic_weights = 1.0 / torch.sqrt(ratio)  # Inverse sqrt of frequency

        # Normalize the dynamic weights to sum to 1
        dynamic_weights /= dynamic_weights.sum()

        # Initialize the final weights for all classes (default to 1.0 for each class)
        num_classes = predict.shape[1]  # Number of output classes
        loss_weights = torch.ones(num_classes, device=predict.device)

        # Combine user-provided class weights (if any) with dynamic weights
        for i, cls in enumerate(unique):
            if self.class_weights is not None:
                # Multiply user-defined class weights with dynamic weights
                loss_weights[cls] = self.class_weights[cls] * dynamic_weights[i]
            else:
                # Use only dynamic weights if no user-provided weights are available
                loss_weights[cls] = dynamic_weights[i]

        # Define the CrossEntropyLoss with the calculated weights
        loss_fn = nn.CrossEntropyLoss(weight=loss_weights, 
                                      ignore_index=self.ignore_index, 
                                      reduction=self.reduction)
        
        # Calculate the loss
        return loss_fn(predict, target)


class AleatoricLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=None, clamp_min=-10, clamp_max=10):
        """
        Aleatoric loss for multi-class segmentation with cross-entropy.
        
        Args:
            reduction (str): 'mean' or 'sum'.
            ignore_index (int or None): Pixel value to ignore in the target mask.
            clamp_min (float): Minimum value for clamping log variance.
            clamp_max (float): Maximum value for clamping log variance.
        """
        super(AleatoricLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, logits, target, log_var):
        """
        Computes aleatoric loss for multi-class segmentation.

        Args:
            logits (Tensor): Model predictions of shape [B, C, H, W].
            target (Tensor): Ground truth labels of shape [B, H, W].
            log_var (Tensor): Predicted log variance of shape [B, C, H, W].

        Returns:
            Tensor: The computed aleatoric loss.
        """
        # Ensure spatial dimensions of logits and target match
        assert logits.shape[2:] == target.shape[1:], "Shape mismatch between logits and target."

        # Compute per-pixel cross-entropy loss without reduction
        cross_entropy = F.cross_entropy(logits, target, reduction='none')

        log_var = torch.clamp(log_var, min=-5, max=5)
        log_var = (log_var - log_var.mean()) / (log_var.std() + 1e-6)

        # Gather log variance for target class
        log_var_target = log_var.gather(1, target.unsqueeze(1)).squeeze(1)  # [B, H, W]

        # Compute precision with a stability term
        precision_target = torch.exp(-log_var_target) + 1e-6

        loss = precision_target * cross_entropy + 0.1 * log_var_target  # [B, H, W]

        # Create a mask to ignore specified pixels if ignore_index is set
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            loss = loss * mask  # Apply mask to the loss

        # Apply the reduction method (mean or sum)
        if self.reduction == 'mean':
            return loss.sum() / mask.sum() if self.ignore_index is not None else loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BalancedCrossEntropyUncertaintyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(BalancedCrossEntropyUncertaintyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predict, target, log_var):
        # Get unique classes and their counts
        unique, unique_counts = torch.unique(target, return_counts=True)
        valid_mask = unique != self.ignore_index
        unique = unique[valid_mask]
        unique_counts = unique_counts[valid_mask]

        # Calculate class weights using inverse square root of frequency
        ratio = unique_counts.float() / torch.numel(target)
        weight = 1.0 / torch.sqrt(ratio)
        weight = weight / torch.sum(weight)

        # Initialize class weights
        loss_weight = torch.ones(predict.shape[1], device=predict.device)
        for i in range(len(unique)):
            loss_weight[unique[i]] = weight[i]

        # Cross-entropy loss without reduction
        cross_entropy_fn = nn.CrossEntropyLoss(weight=loss_weight, 
                                               ignore_index=self.ignore_index, 
                                               reduction='none')
        ce_loss = cross_entropy_fn(predict, target)  # [B, H, W]

        # Clamp and normalize log_var for stability
        log_var = torch.clamp(log_var, min=-5, max=5)
        log_var = (log_var - log_var.mean()) / (log_var.std() + 1e-6)

        # Gather log variance for target class
        log_var_target = log_var.gather(1, target.unsqueeze(1)).squeeze(1)  # [B, H, W]

        # Compute precision with a stability term
        precision_target = torch.exp(-log_var_target) + 1e-6

        # Compute the uncertainty-aware loss
        uncertainty_loss = precision_target * ce_loss + 0.1 * log_var_target  # [B, H, W]

        # Apply the reduction method
        if self.reduction == 'mean':
            return uncertainty_loss.mean()
        elif self.reduction == 'sum':
            return uncertainty_loss.sum()
        else:
            return uncertainty_loss


class FocalLoss(nn.Module):
    """
    Focal Loss with optional alpha weighting per class and support for ignoring an index.
    Args:
        gamma (float): Focusing parameter. Default: 2.0
        alpha (torch.Tensor, optional): Class weights tensor. Shape: [num_classes]
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
        ignore_index (int, optional): Class index to ignore during loss computation.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            # Ensure alpha is a tensor
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # Ensure targets are integers
        targets = targets.long()

        # Move alpha to the same device as logits
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        # Compute log-softmax over the logits
        log_probs = F.log_softmax(logits, dim=1)

        # Gather the log probabilities corresponding to the targets
        targets = targets.squeeze(1)  # Adjust the shape to [B, H, W]
        log_probs = torch.gather(log_probs, dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Apply the ignore_index mask, if provided
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index  # Shape: [B, H, W]
            log_probs = log_probs[valid_mask]
            targets = targets[valid_mask]

        # Compute the focal weight
        probs = log_probs.exp()
        focal_weight = (1 - probs) ** self.gamma
        loss = -focal_weight * log_probs

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Shape: [valid_elements]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
