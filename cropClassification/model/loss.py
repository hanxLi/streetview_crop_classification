import torch
from torch import nn

class BalancedCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss with inverse square root of class frequency weighting.

    Params:
        ignore_index (int): Class index to ignore.
        reduction (str): Reduction method to apply ('mean', 'sum', or 'none').
    '''

    def __init__(self, ignore_index=-100, reduction='mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predict, target):
        # Get unique classes and their counts in the target
        unique, unique_counts = torch.unique(target, return_counts=True)

        # Remove the ignored index from calculations
        valid_mask = unique != self.ignore_index
        unique = unique[valid_mask]
        unique_counts = unique_counts[valid_mask]

        # Calculate class weights using inverse square root of frequency
        ratio = unique_counts.float() / torch.numel(target)  # Frequency ratio
        weight = 1.0 / torch.sqrt(ratio)  # Inverse square root of frequency

        # Normalize the weights to sum to 1 (optional, but common practice)
        weight = weight / torch.sum(weight)

        # Initialize weights for the classes in the prediction
        lossWeight = torch.ones(predict.shape[1], device=predict.device)
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]

        # Use the dynamically calculated weights in CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss(weight=lossWeight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss_fn(predict, target)
