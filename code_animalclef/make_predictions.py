import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wildlife_datasets.datasets import AnimalCLEF2026
import os
import glob
import sklearn.metrics

from paths_and_constants import *
from classify_SeeTurtles import classify_SeeTurtle
from classify_Salamander import classify_Salamander
from classify_Lynx import classify_Lynx
from classify_Lizards import classify_Lizards



# the chosen models (to make feature extraction "on the fly")
feature_model_dict = {'SalamanderID2025': {'name': 'mega384', 'wgt_file': None},
                      'SeaTurtleID2022': {'name': 'mega384', 'wgt_file': None},
                      'LynxID2025': {'name': 'miewid', 'wgt_file': None},
                      'TexasHornedLizards': {'name': 'miewid', 'wgt_file': None}}


# the chosen feature files
feature_file_dict = {'SalamanderID2025': 'SalamanderID2025_Mega-384', #'SalamanderID2025_Mega-384_enh_rfnd',
                     'SeaTurtleID2022': 'SeaTurtleID2022_Mega-384',
                     'LynxID2025': 'LynxID2025_miewid',
                     'TexasHornedLizards': 'TexasHornedLizards_miewid'}


def get_features_labels(dset_name, dset, use_preconpute=True):

    if use_preconpute:
        data = np.load(os.path.join(ROOT_FEATURES, feature_file_dict[dset_name] + '.npz'))
        features, labels = data['all_features'], data['all_labels']
    else:
        assert False
        # TBD extract features and labels

    return features.squeeze(), labels


def collect_test_results(dset, known_labels, pred_labels, submit_df=None):

    if submit_df is None:
        submit_df = pd.DataFrame(columns=['image_id', 'cluster'])

    idxs = np.argwhere(known_labels == -1).flatten().astype(int)
    image_ids = dset.metadata['image_id'][idxs]
    clusters = ['cluster_{}_{}'.format(dset.metadata['dataset'][i], pred_labels[i]) for i in idxs]
    add_df = pd.DataFrame({'image_id': image_ids, 'cluster': clusters})
    submit_df = add_df if submit_df is None else pd.concat((submit_df, add_df))

    return submit_df



# load data
dataset_full = AnimalCLEF2026(ROOT_DATA, transform=None, load_label=True, factorize_label=True, check_files=False)

COMPARE_TO_BASELINE = False
if COMPARE_TO_BASELINE:
    dbg_dict = dict()

# Lynx
dset = dataset_full.get_subset(dataset_full.df['dataset'] == 'LynxID2025')
features, labels = get_features_labels('LynxID2025', dset, use_preconpute=True)
pred_labels, dbg = classify_Lynx(features=features, known_labels=labels, flow=1)
ari_score = sklearn.metrics.adjusted_rand_score(labels[labels > 0], pred_labels[labels > 0])
print('ARI score for {}: {:4.3f}'.format('LynxID2025', ari_score))
submit_df = collect_test_results(dset=dset, known_labels=labels, pred_labels=pred_labels)
#
if COMPARE_TO_BASELINE:
    dbg_dict['LynxID2025'] = {'features': features.squeeze()[labels == -1], 'distances': dbg[0], 'dbscan_result': dbg[1], 'labels': pred_labels[labels == -1]}



# Salamander
dset = dataset_full.get_subset(dataset_full.df['dataset'] == 'SalamanderID2025')
features, labels = get_features_labels('SalamanderID2025', dset, use_preconpute=True)
pred_labels, dbg = classify_Salamander(features=features, known_labels=labels, flow=0)
ari_score = sklearn.metrics.adjusted_rand_score(labels[labels > -1], pred_labels[labels > -1])
print('ARI score for {}: {:4.3f}'.format('SalamanderID2025', ari_score))
submit_df = collect_test_results(dset=dset, known_labels=labels, pred_labels=pred_labels, submit_df=submit_df)
#
if COMPARE_TO_BASELINE:
    dbg_dict['SalamanderID2025'] = {'features': features.squeeze()[labels == -1], 'distances': dbg[0], 'dbscan_result': dbg[1], 'labels': pred_labels[labels == -1]}


# SeaTurtle
dset = dataset_full.get_subset(dataset_full.df['dataset'] == 'SeaTurtleID2022')
features, labels = get_features_labels('SeaTurtleID2022', dset, use_preconpute=True)
pred_labels, dbg = classify_SeeTurtle(features=features, known_labels=labels, flow=0)
ari_score = sklearn.metrics.adjusted_rand_score(labels[labels > 0], pred_labels[labels > 0])
print('ARI score for {}: {:4.3f}'.format('SeaTurtleID2022', ari_score))
submit_df = collect_test_results(dset=dset, known_labels=labels, pred_labels=pred_labels, submit_df=submit_df)
#
if COMPARE_TO_BASELINE:
    dbg_dict['SeaTurtleID2022'] = {'features': features.squeeze()[labels == -1], 'distances': dbg[0], 'dbscan_result': dbg[1], 'labels': pred_labels[labels == -1]}


# TexasHornedLizards
dset = dataset_full.get_subset(dataset_full.df['dataset'] == 'TexasHornedLizards')
features, labels = get_features_labels('TexasHornedLizards', dset, use_preconpute=True)
pred_labels, dbg = classify_Lizards(features=features, known_labels=labels, flow=0)
ari_score = sklearn.metrics.adjusted_rand_score(labels[labels > 0], pred_labels[labels > 0])
print('ARI score for {}: {:4.3f}'.format('TexasHornedLizards', ari_score))
submit_df = collect_test_results(dset=dset, known_labels=labels, pred_labels=pred_labels, submit_df=submit_df)
#
if COMPARE_TO_BASELINE:
    dbg_dict['TexasHornedLizards'] = {'features': features.squeeze(), 'distances': dbg[0], 'dbscan_result': dbg[1], 'labels': pred_labels[labels == -1]}


submit_df.to_csv(os.path.join(ROOT_DEBUG, 'submission.csv'),  index=False)


# debug - compare startup notebook to my flow, in order to align the baseline
if COMPARE_TO_BASELINE:
    with open(os.path.join(ROOT_DEBUG, 'debug_startup_nb'), 'rb') as fd:
        dbg_dict_ref = pickle.load(fd)

    print('comparing to baseline')
    for db_name in dbg_dict_ref.keys():
        ref_data = dbg_dict_ref[db_name]
        local_data = dbg_dict[db_name]
        feat_mismatch = np.abs(local_data['features'] - ref_data['features'])
        print(db_name, '\tmax feature mismatch:', feat_mismatch.max())
        distance_mismatch = np.abs(local_data['distances'] - ref_data['distances'])
        print(db_name, '\tmax distance mismatch:', distance_mismatch.max())
        dbscan_mismatch = np.argwhere(local_data['dbscan_result'] != ref_data['dbscan_result'])
        print(db_name, '\t number of discrepancies in dbscan_result:', dbscan_mismatch.size)
        label_mismatch = np.argwhere(local_data['labels'] != ref_data['labels'])
        print(db_name, '\t number of discrepancies in labels:', label_mismatch.size)

