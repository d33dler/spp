import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cosine_similarity


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

    def _calc_cosine_similarities(self, queries, positives, negatives, av):
        """
        Compute the cosine similarity between query and positive and negatives.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        positives : torch.Tensor
            Tensor of positive embeddings of shape [batch_size, embedding_dim]
        negatives : torch.Tensor
            Tensor of negative embeddings of shape [batch_size, num_negatives, embedding_dim]
        av : int
            Number of augmented views. If av = 0, then just compute the distance without averaging.

        Returns
        -------
        query_pos_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and positive of shape [batch_size,]
        query_neg_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and negatives of shape [batch_size, num_negatives]
        """
        batch_size = queries.size(0)

        # Compute cosine similarity between query and positive
        query_pos_cos_sim = cosine_similarity(queries, positives)

        # Compute cosine similarity between query and negatives
        query_neg_cos_sim = cosine_similarity(queries.unsqueeze(1), negatives, dim=-1).squeeze(1)

        if av > 0:
            # If av > 0, we reshape the cosine similarities for each sample in the batch
            # Then we take the geometric mean across the augmented views
            query_pos_cos_sim = query_pos_cos_sim.view(batch_size // av, av)
            query_neg_cos_sim = query_neg_cos_sim.view(batch_size // av, av, -1)

            query_pos_cos_sim = torch.exp(torch.mean(torch.log(torch.clamp(query_pos_cos_sim, min=1e-8)), dim=1))
            query_neg_cos_sim = torch.exp(torch.mean(torch.log(torch.clamp(query_neg_cos_sim, min=1e-8)), dim=1))

        return query_pos_cos_sim, query_neg_cos_sim

    def forward(self, anchors, positives, negatives, av=0):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        pos_sim, neg_sim = self._calc_cosine_similarities(anchors, positives, negatives, av)
        loss = torch.log1p(torch.sum(torch.exp(neg_sim - pos_sim.unsqueeze(1)), dim=1)).mean() + self.l2_loss(anchors,
                                                                                                              positives)
        return loss

    def l2_loss(self, anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0] + self.l2_reg


class NPairMCLossLSE(NPairMCLoss):
    """
    LSE version Multi-class NPair loss (w/ Log-Sum-Exp for numerical stability)
    based on (K. Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. NIPS 2016)
    """

    def __init__(self, l2_reg=0.02):  # add a regularization coefficient
        super(NPairMCLossLSE, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives, negatives, av=0):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        # Maximum value for stability
        # TODO update or remove
        max_val = torch.max(negatives - positives.unsqueeze(1), dim=1, keepdim=True)[0]
        loss = max_val + torch.log1p(
            torch.sum(torch.exp(negatives - positives.unsqueeze(1) - max_val), dim=1)) + self.l2_loss(
            anchors, positives)

        loss = loss.mean()  # average over the batch
        return loss
