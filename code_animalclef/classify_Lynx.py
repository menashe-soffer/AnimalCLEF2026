from class_utils import *

def classify_Lynx(features, known_labels, flow=0):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)
    known_mask = known_labels > -1

    if flow == 0:
        pred_labels = np.copy(known_labels)
        pred_labels_, dbg = cluster_dbscan(distances=distances[~known_mask][:, ~known_mask], eps=0.3, dbg_sw=True)
        pred_labels[~known_mask] = pred_labels_
        return pred_labels, [distances[~known_mask][:, ~known_mask], dbg]

    if flow == 1:
        pred_labels, eps = cluster_trials(distances=distances, labels=known_labels, try_agl=True)

    return pred_labels, [' ' for l in pred_labels]

