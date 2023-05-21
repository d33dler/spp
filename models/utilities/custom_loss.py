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


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0.02):  # add a regularization coefficient
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, examples):
        """
        examples: 3D tensor of shape (N, 2, d) where d is the dimension of the embedding
        Each row in the first dimension is a pair (anchor, positive example), and there are N such pairs.
        """
        N, _, d = examples.shape

        # ensure examples tensor is floating point and on the correct device
        examples = examples.float().to(examples.device)

        # split the examples into anchors and positive examples
        anchors, positives = torch.split(examples, [1, 1], dim=1)
        anchors = anchors.squeeze(1)
        positives = positives.squeeze(1)

        # compute the similarity with the positive examples
        positive_similarity = F.cosine_similarity(anchors, positives, dim=-1)
        positive_similarity = positive_similarity.unsqueeze(1)

        # compute the similarity with the negative examples
        negative_examples = positives.unsqueeze(0).repeat(N, 1, 1)
        negative_mask = torch.eye(N, device=examples.device).bool()
        negative_examples[negative_mask] = anchors[negative_mask]
        negative_similarity = F.cosine_similarity(anchors.unsqueeze(1), negative_examples, dim=-1)

        # combine the similarities
        logits = torch.cat([positive_similarity, negative_similarity], dim=1)

        # create the labels
        labels = torch.zeros(N, dtype=torch.long, device=examples.device)

        # compute the loss
        loss = F.cross_entropy(logits, labels)

        # L2 regularization
        l2_reg_term = self.l2_reg * torch.norm(examples, p=2)
        loss = loss + l2_reg_term

        return loss


class NPlusOneTupletLoss(nn.Module):
    def __init__(self, topk=5):
        super(NPlusOneTupletLoss, self).__init__()
        self.topk = topk

    def forward(self, positives, negatives):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        # Calculate the loss as per the formula
        max_val = torch.max(negatives - positives.unsqueeze(1), dim=1, keepdim=True)[0]
        loss = max_val + torch.log1p(torch.sum(torch.exp(negatives - positives.unsqueeze(1) - max_val), dim=1))

        loss = loss.mean()  # average over the batch

        return loss
