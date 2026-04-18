import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners


class HybridLoss(torch.nn.Module):

    def __init__(self, s=30.0, m=0.35, triplet_margin=0.4, triplet_weight=0.01):
        super().__init__()
        self.s = s
        self.m = m
        self.triplet_gap = triplet_margin
        self.triplet_weight = triplet_weight
        self.eps = 1e-7

        # This finds the hardest triplets in the current batch
        self.miner = miners.TripletMarginMiner(margin=triplet_margin, type_of_triplets="hard")

        # This calculates the loss based on those mined triplets
        self.triplet_loss_fn = losses.TripletMarginLoss(margin=triplet_margin)



    def forward(self, labels, logits, embeds):
        # --- 1. ArcFace Loss Component ---
        # Logits here are the output of the SubCenterLinear (normalized cosine similarity)
        # We apply the additive angular margin

        # Calculate theta
        theta = torch.acos(torch.clamp(logits, -1.0 + self.eps, 1.0 - self.eps))

        # Target_logits = cos(theta + m)
        target_logits = torch.cos(theta + self.m)

        # Create one-hot mask for the correct classes
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Replace only the 'true' class logits with the margin-shifted version
        output = (one_hot * target_logits) + ((1.0 - one_hot) * logits)
        output *= self.s

        loss_arc = F.cross_entropy(output, labels)

        # Triplet part: The miner finds the indices of the hard samples
        hard_pairs = self.miner(embeds, labels)
        loss_triplet = self.triplet_loss_fn(embeds, labels, hard_pairs)


        # --- 3. Final Weighted Combination ---
        total_loss = loss_arc + (self.triplet_weight * loss_triplet)

        return total_loss, loss_arc, loss_triplet






def calculate_mrr_numpy(features, labels):
    """
    features: numpy array [N, 256]
    labels: numpy array [N]
    """
    # 1. Compute Similarity (Dot product for L2-normalized features)
    sim_matrix = np.dot(features, features.T)

    n = sim_matrix.shape[0]
    # 2. Mask diagonal so an image doesn't match itself
    np.fill_diagonal(sim_matrix, -np.inf)

    # 3. Get ranks for every row
    # argsort gives indices of sorted elements; we flip for descending
    indices = np.argsort(-sim_matrix, axis=1)

    mrr_sum = 0.0
    valid_queries = 0

    for i in range(n):
        target_label = labels[i]
        # Labels of all other images, sorted by similarity to image i
        sorted_labels = labels[indices[i]]

        # Find where the labels match
        matches = np.where(sorted_labels == target_label)[0]

        if len(matches) > 0:
            # Rank is index + 1
            first_match_rank = matches[0] + 1
            mrr_sum += 1.0 / first_match_rank
            valid_queries += 1

    return mrr_sum / valid_queries if valid_queries > 0 else 0.0


# from sklearn.metrics import silhouette_score
#
# # After your DBSCAN/K-Means:
# # features: your [N, 256] embeddings
# # cluster_labels: the output of your clustering algorithm
# score = silhouette_score(distances, cluster_labels, metric='precomputed')
# print(f"Cluster Quality (Silhouette): {score:.4f}")



