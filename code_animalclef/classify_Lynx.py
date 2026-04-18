import os.path
import sklearn.metrics

from class_utils import *
from paths_and_constants import *


def lynx_rsrch_flow(known_labels):

    best_logits = os.path.join(ROOT_MODELS, 'lynx models', 'model 1', 'LynxID2025_resnet.npz')
    best_embeds = os.path.join(ROOT_MODELS, 'lynx models', 'model 1', 'LynxID2025_resnet.npz')

    # read ligits and embeds
    logits = np.load(best_logits)['all_features'].squeeze()
    embeds = np.load(best_embeds)['all_embeddings'].squeeze()

    known_mask = known_labels > -1

    # step 0: baseline clustering
    baseline_embeds = np.load(os.path.join(ROOT_FEATURES, 'LynxID2025_miewid.npz'))['all_features'].squeeze()#embeds#
    _, _, _, _, _, _, _, distances = calc_distances(features=baseline_embeds, labels=known_labels)
    clstr_labels, eps = cluster_trials(distances=distances, labels=known_labels, try_agl=False, eps_range=[0.01, 0.5])

    # step 1: threshold for detection
    dtcts = np.argmax(logits, axis=1)
    dtct_thd = 0.5
    # #
    # Here we theoretically try to find best threshold, ASSUMING prescision on test is identical to precision on train
    #     valid_dtct_mask = (logits.max(axis=1) > dtct_thd) * (dtcts > 0)
    #     dtct_frac_trn = valid_dtct_mask[known_mask].sum() / known_mask.sum()
    #     dtct_frac_tst = valid_dtct_mask[~known_mask].sum() / (~known_mask).sum()
    #     valid_cm = sklearn.metrics.confusion_matrix(known_labels[valid_dtct_mask], dtcts[valid_dtct_mask])
    #     valid_precision = valid_cm.max(axis=1).sum() / valid_cm.sum()
    #     theoretical_good_dtcts = dtct_frac_tst * valid_precision
    #     print('threshold {} :  test  {:5.2f} %   test {:5.2f} %    known precision = {:5.2f}     TGD={}'. \
    #           format(dtct_thd, dtct_frac_trn * 100, dtct_frac_tst * 100, valid_precision * 100, theoretical_good_dtcts))
    #
    # IN PARCTICE, this can't be done, we choose threshold "intuitively"

    dtct_thd = 0.6
    valid_dtct_mask = (logits.max(axis=1) > dtct_thd) * (dtcts > 0)
    test_dtct_mask = valid_dtct_mask * (~known_mask)
    #
    # # try to figure out test_detects vs. test_clusters
    # test_dtcts = dtcts[~known_mask]
    # test_clstrs = clstr_labels[~known_mask]
    # groups = []
    # open_group = np.unique(dtcts[(~valid_dtct_mask) * (~known_mask)])
    # for id_idx, dtct_id in enumerate(np.unique(dtcts[test_dtct_mask])):
    #     mask = test_dtcts == dtct_id
    #     tst_labels = test_clstrs[mask]
    #     groups.append(np.unique(tst_labels))
    #     print('\n', id_idx, dtct_id, ':', groups[-1])
    # #
    # m = len(groups)
    # intesecting_groups = []
    # mix_table = np.zeros((m, m+1), dtype=int)
    # for i_grp_1 in range(m):
    #     for i_grp_2 in range(m):
    #         if i_grp_1 != i_grp_2:
    #             print(set(groups[i_grp_1]).intersection(set(groups[i_grp_2])))
    #         mix_table[i_grp_1][i_grp_2] = len(set(groups[i_grp_1]).intersection(set(groups[i_grp_2])))
    #         mix_table[i_grp_1][-1] = len(set(groups[i_grp_2]).intersection(set(open_group)))
    # print(mix_table)

    dtcts_ovrd = dtcts + clstr_labels.max() + 1
    labels = np.copy(clstr_labels)
    labels[valid_dtct_mask] = dtcts_ovrd[valid_dtct_mask]

    #
    # labels = np.copy(known_labels)
    # labels[test_dtct_mask] = dtcts[test_dtct_mask]
    # start_for_clusterin = labels.max() + 1
    #
    # for_clusterring_mask = (~known_mask) * (~test_dtct_mask)
    # _, _, _, _, _, _, _, distances = calc_distances(embeds[~known_mask], labels[~known_mask], metric='similarity')
    #
    # cluster_labels, eps = cluster_trials(distances=distances, labels=labels[~known_mask], try_agl=False, eps_range=[0.01, 0.5])
    # labels[for_clusterring_mask] = cluster_labels[~test_dtct_mask[~known_mask]] + start_for_clusterin


    return labels







def classify_Lynx(features, known_labels, flow=0):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)
    known_mask = known_labels > -1

    if flow == 0:
        pred_labels = np.copy(known_labels)
        pred_labels_, dbg = cluster_dbscan(distances=distances[~known_mask][:, ~known_mask], eps=0.3, dbg_sw=True)
        pred_labels[~known_mask] = pred_labels_
        print('found {} distinct IDs'.format(len(np.unique(pred_labels_))))
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

    if flow == 3:
        pred_labels = lynx_rsrch_flow(known_labels)
        # pred_labels = np.copy(known_labels)
        # pred_labels_, dbg = cluster_dbscan(distances=distances[~known_mask][:, ~known_mask], eps=0.11, dbg_sw=True)
        # pred_labels[~known_mask] = pred_labels_
        # return pred_labels, [distances[~known_mask][:, ~known_mask], dbg]

    return pred_labels, [' ' for l in pred_labels]

