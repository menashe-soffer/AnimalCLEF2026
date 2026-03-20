from pyexpat import features

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
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


def cluster(distances):

    pass


def classify_using_knowns(distances, labels, n=3, weighted=False):

    idxs_to_classify = np.argwhere(labels == -1).flatten()
    known_label_dictionary = np.unique(labels[labels > -1])
    output_labels = np.copy(labels)

    for qidx in idxs_to_classify:
        best_dist, best_dist_id = 9, -1
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
                weighting_distance = 1
            weighted_distance = avg_distances / weighting_distance
            if avg_distances < best_dist:
                best_dist, best_dist_id = avg_distances, id
                #print('\tunweighted:\t', best_dist, best_dist_id)
            if weighted_distance < best_wgted_dist:
                best_wgted_dist, best_wgted_dist_id = weighted_distance, id
                #print('\tweighted:\t', best_dist, best_dist_id)

        # TDB: open set, store weighting anyway and set threshold for outlayer
        output_labels[qidx] = best_wgted_dist_id if weighted else best_dist_id

        # TBD: cluster outlayers

    return output_labels





def map_clusters_and_labels(known_labels, clustered_labels):

    pass


if __name__ == '__main__':

    ROOT_FOLDER = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026'

