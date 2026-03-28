import numpy as np


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



