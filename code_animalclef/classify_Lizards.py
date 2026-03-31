import numpy as np

from class_utils import *

def classify_Lizards(features, known_labels, flow=0):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)

    if flow == 0:
        pred_labels, dbg = cluster_dbscan(distances=distances, eps=0.24, dbg_sw=True)

    return pred_labels, dbg

