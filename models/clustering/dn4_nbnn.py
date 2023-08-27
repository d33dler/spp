from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
import torch.nn.functional as F


# Deep Nearest Neighbor Neural Network (DN4) NBNN implementation

class I2C_KNN(nn.Module):
    """
    KNN-I2C (Image to class) metric
    :param k_neighbors: number of nearest neighbors
    :param classes: number of classes
    :param strategy: strategy to use for computing the similarity between query and support set
    :param av_num: number of augmented views per sample
    :param apn: whether to use APN (Anchor Positive Negative) similarity
    :param shots: number of shots per class
    :param targets: target classes
    :param kwargs: additional arguments
    """

    def __init__(self, k_neighbors: int, classes=5, aggregation='gmean'):
        super(I2C_KNN, self).__init__()
        self.shots = None
        self.neighbor_k = k_neighbors
        self.classes = classes
        self.strategy_map = {
            "1:1": self._strategy_ovo,
            "N:1": self._strategy_rvo,
            "N:N": self._strategy_rvr,
        }
        self.agg_map = {
            'gmean': self._geometric_mean,
            'softmax': self._softmax_mean,
            'hmean': None
        }
        self.aggregation = self.agg_map[aggregation] if aggregation in self.agg_map else self._geometric_mean

    @staticmethod
    def _geometric_mean(t: Tensor, dim=0) -> Tensor:
        geom_mean = torch.exp(torch.mean(torch.log(torch.clamp_min(t, 1e-8)), dim=dim, keepdim=False))
        return geom_mean

    def _softmax_mean(self, t: Tensor, dim=0) -> Tensor:
        # use torch functional to compute softmax across dim
        # multiply the softmax probabilities with the tensor and compute arithmetic mean
        softmax = F.softmax(t, dim=dim)
        mean = torch.mean(softmax * t, dim=dim, keepdim=False)
        return mean

    @staticmethod
    def normalize_tensor(tensor: Tensor, dim: int) -> Tensor:
        tensor_norm = torch.norm(tensor, 2, dim, True)
        return tensor / tensor_norm

    @staticmethod
    def get_cosine_similarity(query: Tensor, support: Tensor) -> Tensor:
        return query @ support

    def get_topk_cosine_sum(self, matrix: Tensor, split_dim=1, topk_dim: int = 1) -> Tensor:
        matrix_split = torch.split(matrix, 441, dim=split_dim)
        return torch.stack([torch.sum(torch.topk(v, self.neighbor_k, dim=topk_dim)[0]) for v in matrix_split])

    @staticmethod
    def get_topk_values(matrix: Tensor, k: int, dim: int) -> Tensor:
        return torch.topk(matrix, k, dim)[0]

    def apply_geometric_mean(self, tensor: Tensor, av_num: int) -> Tensor:
        return self._geometric_mean(torch.clamp_min(tensor, 1e-8)) if av_num > 1 else tensor

    def visualize_similarity_matrices(self, similarity_matrices):
        # Ensure the tensor is on CPU and convert it to numpy
        similarity_matrices_np = similarity_matrices.cpu().detach().numpy()
        # Get the batch size and number of classes
        batch_size = similarity_matrices_np.shape[0]
        num_classes = similarity_matrices_np.shape[2]

        # Loop over each item in the batch
        for i in range(batch_size):
            # Create a new figure
            fig, axs = plt.subplots(1, num_classes, figsize=(15, 15))

            # Loop over each class
            for j in range(num_classes):
                # Create the heatmap in the subplot
                axs[j].imshow(similarity_matrices_np[i, :, j, :], cmap='bwr', vmin=0.9, vmax=1)

                # Optionally, you can add labels, title, etc.
                axs[j].set_title(f"Class {j + 1}")

            # Add a colorbar to the right
            fig.colorbar(plt.cm.ScalarMappable(cmap='bwr'), ax=axs.ravel().tolist())

            # Save the figure to a file
            plt.savefig(f"tmp/dn4_csim_heatmaps/heatmap_batch_{i + 1}.png")

            # Close the figure to free up memory
            plt.close(fig)
        exit(0)

    def cosine_similarity(self, anchor: Tensor, support_set: Tensor, av_num: int = 1, sav_num: int= 1, strategy=None, **kwargs) -> Tensor:
        """
        Compute cosine similarity between query and support set
        :param anchor: query tensor
        :param support_set: support set tensor
        :param av_num: number of augmented views per sample
        :param kwargs: additional arguments
        :return: cosine similarity tensor
        """

        # Reshape and permute query and support set tensors
        anchor = self.l2_norm(anchor)
        support_set = self.l2_norm(support_set)  # (L * S * AV) * 64 * (H*W)

        S = support_set.contiguous().view(1, -1, support_set.size(2))  # 1 * (B * AV * (H*W)) * 64
        S = S.permute(0, 2, 1)  # (B * AV) * x * 64
        # Compute cosine similarity between query and support set

        innerprod_mx = torch.matmul(anchor.unsqueeze(1), S)
        # print("inner*: ", innerprod_mx.size())

        # Reshape innerproduct into augmented views sets of each query
        B, *_ = innerprod_mx.size()
        innerprod_mx = innerprod_mx.squeeze()
        innerprod_mx = innerprod_mx.view(B // av_num, av_num, innerprod_mx.size(1),
                                         self.classes, innerprod_mx.size(2) // self.classes)
        # print("inner: ", innerprod_mx.size())
        # innerprod_mx = self.rbf_kernel(innerprod_mx)
        apn = None
        if kwargs.get('apn', False):
            if av_num > 1:
                raise 'AngularLoss APN not supported for multi AVs per sample'
            # Compute Anchor+Positive to Negative similarity
            targets = kwargs.get('targets')
            shots = self.shots
            positives = support_set.view(self.classes,
                                         shots,
                                         support_set.size(1),
                                         support_set.size(2))[targets]
            ap = anchor.unsqueeze(1) + positives
            # l2 normalize
            ap = ap / torch.norm(ap, 2, 3, True)
            ap = ap.flatten(start_dim=1, end_dim=2)
            # extract negative samples
            support_negatives = self.extract_negative_samples(support_set, targets, self.classes, shots)
            support_negatives = support_negatives.flatten(start_dim=1, end_dim=2).permute(0, 2, 1)

            # compute inner product with negative samples
            apn = torch.bmm(ap, support_negatives)
            apn = apn.view(apn.size(0), apn.size(1), self.classes - 1 , apn.size(2) // (self.classes-1))
            # print(apn.size())
            # print('apn :', apn.size())
            topk_apn_value, _ = torch.topk(apn, self.neighbor_k, -1)
            # Compute image-to-class similarity
            # print('topk:', topk_apn_value.size())
            apn = torch.clamp(torch.sum(torch.sum(topk_apn_value, -1), -2), min=1e-8)
        # Choose the top-k nearest neighbors
        if strategy == 'N:N' and sav_num > 1:
            B, AV, HW, L, C = innerprod_mx.size()
            topk_value, _ = torch.topk(innerprod_mx.reshape( B, AV, HW, L, sav_num, C // sav_num), self.neighbor_k, -1)
            img2class_sim = torch.sum(torch.sum(topk_value, -1), -3)
            img2class_sim = torch.clamp(img2class_sim, min=1e-8)
            img2class_sim = self.aggregation(img2class_sim, dim=-1)
        else:
            topk_value, _ = torch.topk(innerprod_mx, self.neighbor_k, -1)
            # Compute image-to-class similarity
            img2class_sim = self._compute_img2class_sim(topk_value, **kwargs)
        # Aggregate the similarity values of all augmented views of each query
        similarity_ls = self.aggregation(img2class_sim, dim=1) if img2class_sim.size(1) > 1 else img2class_sim.squeeze(
            1)
        return similarity_ls

    def extract_negative_samples(self, support_set, target_indices, L, S) -> Tensor:
        negative_samples_list = []

        # Iterate through the target indices and extract the negative samples
        for target_index in target_indices:
            mask = torch.ones(L * S, dtype=torch.bool)
            mask[target_index * S: (target_index + 1) * S] = False
            negative_samples_for_target = support_set[mask]
            negative_samples_list.append(negative_samples_for_target)

        # Stack the negative samples into a tensor
        negative_samples_tensor = torch.stack(negative_samples_list)

        return negative_samples_tensor

    def l2_norm(self, x: Tensor) -> Tensor:
        x = x.contiguous().view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x_norm = torch.norm(x, 2, 2, True)
        return x / x_norm

    def _compute_img2class_sim(self, topk_value: Tensor, **kwargs) -> Tensor:
        if self.neighbor_k > 1:
            img2class_sim = torch.sum(torch.sum(topk_value, -1), -2)
        else:
            img2class_sim = torch.sum(topk_value.squeeze(-1), -2)
        return img2class_sim

    def _strategy_ovo(self, q, S, qAV_num=1, SAV_num=1, i=0):
        """
        1:1 strategy : 1 query AV, 1 support class AV-subset
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class

        """
        inner_sim = torch.zeros(qAV_num, len(S) // SAV_num).cuda()
        for j in range(0, len(S), qAV_num):
            cls_ix = j // SAV_num
            for av in range(qAV_num):
                support_set_sam = self.normalize_tensor(S[j + av], 0)  # support set AV

                query_sam = self.normalize_tensor(q[i + av], 1)  # query sample AV
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)

                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[av, cls_ix] = torch.sum(topk)
        return inner_sim

    def _strategy_rvo(self, q, S, qAV_num=1, SAV_num=None, i=0):
        """
        N:1 strategy : N query AVs, 1 support clas
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: ignored, used for consistency
        param i: index of query sample
        """
        inner_sim = torch.zeros(qAV_num, len(S)).cuda()
        for j in range(len(S)):
            support_set_sam = self.normalize_tensor(S[j], 0)  # support set AV

            for av in range(qAV_num):
                query_sam = self.normalize_tensor(q[i + av], 1)  # query sample AV
                # cosine similarity between a query sample and a support category
                innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                topk = self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1)
                inner_sim[av, j] = torch.sum(topk)
        return inner_sim

    def _strategy_rvr(self, q, S, qAV_num=1, SAV_num=1, i=0):
        """
        N:N strategy : N query AVs vs N support AVs
        param q: query sample
        param S: support set
        param qAV_num: number of query AV-samples per episode
        param SAV_num: number of support class AV-subsets per class
        param i: index of query sample
        """
        cls_size = len(S) // SAV_num
        inner_sim = torch.zeros(qAV_num * SAV_num, cls_size).cuda()
        index = [0 for _ in range(cls_size)]
        for k in range(qAV_num):

            query_sam = self.normalize_tensor(q[i + k], 1)  # query sample AV
            for j in range(0, len(S), SAV_num):
                cls_ix = j // SAV_num
                for av in range(SAV_num):
                    support_set_sam = self.normalize_tensor(S[j + av], 0)
                    # cosine similarity between a query sample and a support category

                    innerproduct_matrix = self.get_cosine_similarity(query_sam, support_set_sam)
                    inner_sim[index[cls_ix], cls_ix] = torch.sum(
                        self.get_topk_values(innerproduct_matrix, self.neighbor_k, 1))
                    index[cls_ix] += 1
        return inner_sim

    def forward(self, q, S, av_num=1, SAV_num=1, strategy='N:1', shot_num=1, **kwargs):
        strategy = strategy if strategy is not None else 'N:1'
        self.shots = shot_num
        self.classes = len(S) // (SAV_num * shot_num)
        return self.cosine_similarity(q, S, av_num, SAV_num, strategy=strategy, **kwargs)


class I2C_KNN_AM(I2C_KNN):

    def __init__(self, neighbor_k, classes=5, _attention_func=None):
        super().__init__(neighbor_k, classes)
        self._attention_func = _attention_func

    def _compute_img2class_sim(self, topk_value, **kwargs):
        """
        Compute image to class similarity using M(x) Class-related Attention Mechanism
        :param topk_value: topk values
        :return: image to class similarity
        """
        x = kwargs['x']
        av = kwargs['av']
        att_x = kwargs['attention_map']  # (B * AV, 1, H, W)
        att_x = att_x.reshape(-1, att_x.size(2)* att_x.size(3), 1) # (B * AV, H, W, 1)
        if self.neighbor_k == 1:
            img2class_sim = topk_value
        else:
            img2class_sim = torch.sum(topk_value, -1)  # (B, AV, HW, L)
        img2class_sim = img2class_sim.reshape(-1, img2class_sim.size(2), img2class_sim.size(3))  # (B * AV, HW, L)
        img2class_sim = torch.sum(att_x * img2class_sim, -2)  # (B * AV, 1, L)
        # print(img2class_sim.size())

        img2class_sim = img2class_sim.reshape(-1, av, img2class_sim.size(1))
        return img2class_sim

    def forward(self, q, S, av_num=1, SAV_num=1, strategy='N:1', shot_num=1, **kwargs):
        return self.cosine_similarity(q, S, av_num, x=q, av=av_num, **kwargs)
