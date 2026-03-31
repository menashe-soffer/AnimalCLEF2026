from pyexpat import features
from turtledemo.forest import start

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.cluster
from sklearn.metrics import adjusted_rand_score
import os

import gc
import psutil

def print_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Convert bytes to Gigabytes
    gb_usage = mem_info.rss / (1024 ** 3)
    print(f"Current RAM Usage: {gb_usage:.2f} GB")


def calc_distances(features, labels, metric='similarity'):

    tst_mask = labels == -1
    trn_mask = labels > -1

    features_norm = sklearn.preprocessing.normalize(features.squeeze(), norm='l2', axis=1)
    similarity = features_norm @ features_norm.T

    if metric == 'similarity':
        distances = (np.max(similarity) - np.maximum(similarity, 0)) / np.max(similarity)

    if metric == 'euclidian':
        ab = np.diag(similarity)
        distances = np.sqrt(ab + ab.T - 2 * similarity)

    trn_labels = labels[trn_mask]
    tst_labels = labels[tst_mask]
    trn_fetures = features[trn_mask]
    tst_features = features[tst_mask]
    distances_trn_trn = distances[trn_mask][:, trn_mask]
    distances_tst_trn = distances[tst_mask][:, trn_mask]
    distances_tst_tst = distances[tst_mask][:, tst_mask]

    return trn_labels, tst_labels, trn_fetures, tst_features, distances_trn_trn, distances_tst_trn, distances_tst_tst, distances


def cluster_dbscan(distances, eps=None, dbg_sw=False):

    def cluster_step(distances, eps):

        clustering = sklearn.cluster.DBSCAN(eps=eps, metric='precomputed', min_samples=2)
        clusters = clustering.fit(distances)

        labels = np.array(clusters.labels_)
        neg_indices = np.where(labels == -1)[0]
        new_labels = np.arange(labels.max() + 1, labels.max() + 1 + len(neg_indices))
        labels[neg_indices] = new_labels

        return labels, np.array(clusters.labels_)


    if eps is None:
        eps_vec, ncluster_vec = [], []
        for eps in np.arange(start=0.2, stop=0.7, step=0.1):
            labels, dbg = cluster_step(distances, eps)
            eps_vec.append(eps)
            ncluster_vec.append(np.unique(labels).size)

        plt.plot(eps_vec, ncluster_vec)
        plt.grid(True)
        plt.xlabel('eps')
        plt.ylabel('num clusters')
        plt.show()

    else:
        labels, dbg = cluster_step(distances, eps)
        if dbg_sw:
            return labels, dbg
        else:
            return labels


def cluster_agglomerative(distances, threshold=None):

    def cluster_step(distances, thresh):
        # n_clusters must be None to use distance_thresholdfrom sklearn.cluster import DBSCANfrom sklearn.cluster import DBSCAN


        clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thresh,
            metric='precomputed',
            linkage='complete' # 'complete' or 'average' are best for Re-ID
        )
        clusters = clustering.fit(distances)
        return clusters.labels_

    if threshold is None:
        # Scanning the distance threshold (similar to your eps scan)
        thresh_vec, n_clusters_vec = [], []
        # Since your Triplet Margin was 0.3-0.45, scan around that range
        for t in np.arange(start=0.1, stop=1.0, step=0.05):
            labels = cluster_step(distances, t)
            thresh_vec.append(t)
            n_clusters_vec.append(np.unique(labels).size)

        plt.figure(figsize=(8, 5))
        plt.plot(thresh_vec, n_clusters_vec, marker='o')
        plt.title('Agglomerative Clustering: Threshold Scan')
        plt.grid(True)
        plt.xlabel('distance_threshold')
        plt.ylabel('Number of Clusters')
        plt.show()
    else:
        return cluster_step(distances, threshold)

def fill_outlayers(labels):

    mask = labels == -1
    new_ids = np.arange(mask.sum()) + labels.max() + 1
    labels[mask] = new_ids

    return labels


def cluster_trials(distances, labels, try_agl=False):

    # try both dbscan and agglomerative, scan parameters, and chose the one that gives best ARI for known labels
    # unknown labels are -1

    know_mask = labels > -1
    assert know_mask.sum() > 0

    best_ari_score = -1

    for eps in np.arange(start=0.1, stop=0.5, step=0.01):
        for min_samples in [2, 3, 4]:

            clustering = sklearn.cluster.DBSCAN(eps=eps, metric='precomputed', min_samples=min_samples)
            clusters = clustering.fit(distances)

            pred_labels = np.array(clusters.labels_)
            pred_labels = fill_outlayers(pred_labels)

            ari_score = adjusted_rand_score(labels[know_mask], pred_labels[know_mask])
            #print(eps, ari_score)

            if ari_score > best_ari_score:

                best_ari_score = ari_score
                best_min_samples = min_samples
                best_eps = eps
                best_pred_labels = np.copy(pred_labels)

    print('for dbscan:    best min_samples={}    best eps={}    best ARI score={:4.3f}'.format(best_min_samples, best_eps, best_ari_score))

    if try_agl:
        best_ari_score_agl = -1
        num_known_ids = np.unique(labels[know_mask]).size
        for linkage in ['ward', 'complete', 'average', 'single'][1:]:
            best_ari_in_linkage = -1
            n_trial_list = np.linspace(start=num_known_ids, stop=num_known_ids+(labels == -1).sum(), num=10).astype(int)
            for n_clusters in n_trial_list:
                agl = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree='auto')
                pred_labels = agl.fit_predict(distances)
                ari_score = adjusted_rand_score(labels[know_mask], pred_labels[know_mask])
                #print(linkage, n_clusters, ari_score)
                if ari_score > best_ari_in_linkage:
                    best_ari_in_linkage = ari_score
                    best_n = n_clusters
            #print(linkage, best_n, best_ari_in_linkage)
            # fine tune
            low = np.max(n_trial_list[n_trial_list < best_n]) if best_n > n_trial_list.min() else best_n
            high = np.min(n_trial_list[n_trial_list > best_n]) if best_n < n_trial_list.max() else best_n
            n_trial_list = np.linspace(start=low, stop=high, num=10).astype(int)
            for n_clusters in n_trial_list:
                agl = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree='auto')
                pred_labels = agl.fit_predict(distances)
                ari_score = adjusted_rand_score(labels[know_mask], pred_labels[know_mask])
                #print(linkage, n_clusters, ari_score)
                if ari_score > best_ari_in_linkage:
                    best_ari_in_linkage = ari_score
                    best_n = n_clusters
                if ari_score > best_ari_score_agl:
                    best_ari_score_agl = ari_score
                    best_linkage = linkage
                    best_n = n_clusters
                    best_pred_labels_agl = np.copy(pred_labels)
        print(best_linkage, best_n, best_ari_score)

        if best_ari_score_agl > best_ari_score:
            best_pred_labels = best_pred_labels_agl

    return best_pred_labels, best_eps


def classify_using_knowns(distances, labels, n=3, weighted=False):

    idxs_to_classify = np.argwhere(labels == -1).flatten()
    known_label_dictionary = np.unique(labels[labels > -1])
    output_labels = np.copy(labels)
    dbg_strs = ['' for i in range(len(labels))]
    best_distances_vec, best_wgt_vec = np.zeros_like(labels, dtype=float), np.zeros_like(labels, dtype=float)

    for qidx in idxs_to_classify:
        best_dist, best_dist_id, wgt_for_best = 9, -1, 0
        best_wgted_dist, best_wgted_dist_id = 9, -1
        #print('qidx = ', qidx)
        for id in known_label_dictionary:
            sample_idxs = np.argwhere(id == labels).flatten()
            row_distances = distances[qidx, sample_idxs]
            pick_distances_idxs = np.argsort(row_distances)[:n]
            pick_distances = row_distances[pick_distances_idxs]
            close_subset_idxs = sample_idxs[pick_distances_idxs]
            close_subset_distance_matrix = distances[close_subset_idxs][:, close_subset_idxs]
            cls_m = close_subset_idxs.size
            avg_distances = pick_distances.mean()
            if cls_m > 1:
                weighting_distance = close_subset_distance_matrix[~ np.eye(cls_m, dtype=bool)].reshape((cls_m, cls_m-1)).min(axis=1).mean()
            else:
                weighting_distance = -1
            weighted_distance = avg_distances / weighting_distance
            #
            fcls_m = (labels == id).sum()
            if fcls_m > 1:
                full_clstr_dist_mat = distances[labels==id][:, labels==id]
                full_clstr_dist_mat = full_clstr_dist_mat[~np.eye(fcls_m, dtype=bool)].reshape(fcls_m, fcls_m - 1)
                full_clstr_wgt = full_clstr_dist_mat.min(axis=1).mean()
            else:
                full_clstr_wgt = -1

            if avg_distances < best_dist:
                best_dist, best_dist_id, wgt_for_best = avg_distances, id, full_clstr_wgt
                #print('\tunweighted:\t', best_dist, best_dist_id)
            if weighted_distance < best_wgted_dist:
                best_wgted_dist, best_wgted_dist_id = weighted_distance, id
                #print('\tweighted:\t', best_dist, best_dist_id)

        # TDB: open set, store weighting anyway and set threshold for outlayer
        output_labels[qidx] = best_wgted_dist_id if weighted else best_dist_id
        dbg_strs[qidx] = '{:3.2f} / {:3.2f}'.format(best_dist, wgt_for_best)
        best_distances_vec[qidx], best_wgt_vec[qidx] = best_dist, wgt_for_best

    # go over all predictions, and turn to outlier according to threshold
    thds = np.zeros_like(labels, dtype=float)
    thds[idxs_to_classify] = np.minimum(0.48, np.maximum(0.24, best_wgt_vec[idxs_to_classify] * 1.5))
    outlier_mask = np.zeros_like(labels, dtype=bool)
    outlier_mask[idxs_to_classify] = best_distances_vec[idxs_to_classify] > thds[idxs_to_classify]
    output_labels[outlier_mask] = -1

    return output_labels, dbg_strs





def map_clusters_and_labels(known_labels, clustered_labels):

    pass


if __name__ == '__main__':

    ROOT_FOLDER = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026'

