import numpy as np
from class_utils import *


def classify_SeeTurtle(features, known_labels):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)
    labels, dbg_strs = classify_using_knowns(distances=distances, labels=known_labels)
    not_classified_mask = labels == -1
    #cluster(distances=distances[not_classified_mask][:, not_classified_mask])
    clabels = cluster(distances=distances[not_classified_mask][:, not_classified_mask], eps=0.55)
    start_new_label = labels.max() + 1
    print('new clusters generated from label', start_new_label)
    clabels += start_new_label
    labels[not_classified_mask] = clabels

    return labels, dbg_strs