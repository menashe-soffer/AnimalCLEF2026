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
        pred_labels, eps = cluster_trials(distances=distances, labels=known_labels, try_agl=False, eps_range=[0.01, 0.5])

    if flow == 2:
        pred_labels, eps = cluster_trials(distances=distances, labels=known_labels, try_agl=False, eps_range=[0.001, 0.05], min_samples_list=[10])
        big_ids = np.argwhere(np.bincount(pred_labels) > 100).flatten()
        mask = np.array([p in big_ids for p in pred_labels], dtype=bool)
        sub_distances = distances[~mask][:, ~mask]
        sub_pred_labels, eps = cluster_trials(distances=sub_distances, labels=known_labels[~mask], try_agl=True, eps_range=[0.01, 0.5])
        pred_labels[~mask] = sub_pred_labels


    return pred_labels, [' ' for l in pred_labels]

