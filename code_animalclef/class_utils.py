from pyexpat import features
from turtledemo.forest import start

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import sklearn.cluster
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


def cluster(distances, eps=None):

    def cluster_step(distances, eps):

        clustering = sklearn.cluster.DBSCAN(eps=eps, metric='precomputed', min_samples=2)
        clusters = clustering.fit(distances)

        labels = np.array(clusters.labels_)
        neg_indices = np.where(labels == -1)[0]
        new_labels = np.arange(labels.max() + 1, labels.max() + 1 + len(neg_indices))
        labels[neg_indices] = new_labels

        return labels


    if eps is None:
        eps_vec, ncluster_vec = [], []
        for eps in np.arange(start=0.2, stop=0.7, step=0.1):
            labels = cluster_step(distances, eps)
            eps_vec.append(eps)
            ncluster_vec.append(np.unique(labels).size)

        plt.plot(eps_vec, ncluster_vec)
        plt.grid(True)
        plt.xlabel('eps')
        plt.ylabel('num clusters')
        plt.show()

    else:

        return cluster_step(distances, eps)




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

