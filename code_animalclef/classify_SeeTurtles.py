import numpy as np
from class_utils import *


def classify_SeeTurtle(features, known_labels, flow=1):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)

    known_mask = known_labels > -1

    if flow in [0, 1]:
        pred_labels = np.copy(known_labels)
        if flow == 0:
            pred_labels = np.copy(known_labels)
            pred_labels_, dbg = cluster_dbscan(distances=distances[~known_mask][:, ~known_mask], eps=0.4, dbg_sw=True)
            pred_labels[~known_mask] = pred_labels_
            return pred_labels, [distances[~known_mask][:, ~known_mask], dbg]

        if flow == 1:
            # tmp_pred_labels, _ = cluster_trials(distances=distances, labels=known_labels)
            # pred_labels[~known_mask] = tmp_pred_labels[~known_mask]
            pred_labels, _ = cluster_trials(distances=distances, labels=known_labels)

        return pred_labels, [' ' for l in pred_labels]


    if flow in [2, 3]:
        pred_labels, eps = cluster_trials(distances=distances, labels=known_labels, try_agl=flow == 4)

    if flow == 3:
        return pred_labels, [' ' for l in pred_labels]

    if flow == 2:
        labels, dbg_strs = classify_using_knowns(distances=distances, labels=known_labels)
        #assert False
        not_classified_mask = labels == -1
        #cluster(distances=distances[not_classified_mask][:, not_classified_mask])
        clabels = cluster_dbscan(distances=distances[not_classified_mask][:, not_classified_mask], eps=eps)
        start_new_label = labels.max() + 1
        print('new clusters generated from label', start_new_label)
        clabels += start_new_label
        labels[not_classified_mask] = clabels
        #

        return labels, dbg_strs