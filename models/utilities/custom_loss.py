import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginWithDistanceLossImageToClass(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super(TripletMarginWithDistanceLossImageToClass, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, distances_batch, target_batch):
        losses = []

        for distances, target in zip(distances_batch, target_batch):
            positive_class = target.item()
            positive_dist = distances[positive_class]

            negative_dists = torch.cat((distances[:positive_class], distances[positive_class + 1:]))
            negative_dist, _ = torch.min(negative_dists, dim=0)

            loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)
            losses.append(loss)

        losses = torch.stack(losses)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class NPairMCLoss(nn.Module):
    """
    Original Multi-class NPair loss
    (K. Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. NIPS 2016)
    """

    def __init__(self, l2_reg=0.02):  # add a regularization coefficient

        super(NPairMCLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, positives, negatives):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        # Calculate the loss as per the formula
        loss = torch.log1p(torch.sum(torch.exp(negatives - positives.unsqueeze(1)), dim=1)).mean()
        return loss


class NPairMCLossLSE(nn.Module):
    """
    LSE version Multi-class NPair loss (w/ Log-Sum-Exp for numerical stability)
    based on (K. Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. NIPS 2016)
    """

    def __init__(self, l2_reg=0.02):  # add a regularization coefficient
        super(NPairMCLossLSE, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, positives, negatives):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        # Maximum value for stability
        max_val = torch.max(positives, torch.max(negatives, dim=1)[0])

        # Shift the values of positives and negatives for stability
        positives_shifted = positives - max_val
        negatives_shifted = negatives - max_val.unsqueeze(1)

        # Calculate the loss as per the formula
        loss = max_val + torch.log1p(
            torch.sum(torch.exp(negatives_shifted - positives_shifted.unsqueeze(1)), dim=1)).mean()

        return loss
