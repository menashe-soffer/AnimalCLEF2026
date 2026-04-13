from pyexpat import features

import numpy as np
import matplotlib
from IPython.core.display import display_html
from IPython.core.pylabtools import figsize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import os

import gc
import psutil
from tqdm import tqdm

from wildlife_datasets.datasets import AnimalCLEF2026


from paths_and_constants import *

from class_utils import calc_distances



def print_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Convert bytes to Gigabytes
    gb_usage = mem_info.rss / (1024 ** 3)
    print(f"Current RAM Usage: {gb_usage:.2f} GB")




def visualize_distances(feat_version_names, feat_versions, labels, mark_labels=None):

    def evaluate_contrast(distances, labels):

        inner_sum, inner_cnt, outer_sum, outer_cnt = 0, 0, 0, 0
        groups = np.unique(labels)
        for grp in groups:
            mask = labels == grp
            inside_dist = distances[mask][:, mask].min(axis=1)
            inner_sum += inside_dist.sum()
            inner_cnt += inside_dist.size
            outer_dist = distances[mask][:, ~mask].min(axis=1)
            outer_sum += outer_dist.sum()
            outer_cnt += outer_dist.size

        inner_avg = inner_sum / inner_cnt
        outer_avg = outer_sum / outer_cnt

        return 1 - inner_avg / outer_avg, inner_avg, outer_avg


    num_versions = len(feat_versions)
    plot_rows = int(np.ceil(np.sqrt(num_versions)))
    plot_cols = int(np.ceil(num_versions / plot_rows))

    fig, ax = plt.subplots(plot_rows, 3 * plot_cols, figsize=(16, 12), sharex=False, sharey=False)
    ax1 = ax[:, :plot_cols] if plot_rows > 1 else ax[:plot_cols]
    ax2 = ax[:, plot_cols:] if plot_rows > 1 else ax[plot_cols:]

    display_distances, display_hists, trn_label_list = [], [], []
    for i in range(num_versions):
        #print_memory()
        trn_labels, _, _, _, distances_trn_trn, _, _, _ = calc_distances(feat_versions[i], labels, metric='similarity')
        good_labels = np.argwhere(np.bincount(trn_labels) > 6).flatten()
        mask = np.array([l in good_labels for l in trn_labels])
        trn_labels_ = trn_labels[mask]
        distances_trn_trn_ = distances_trn_trn[mask][:, mask]
        #print_memory()
        reorder = np.argsort(trn_labels_)
        distances_trn_trn_ = distances_trn_trn_[reorder][:, reorder]
        trn_labels_ = trn_labels_[reorder]
        trn_label_list.append(trn_labels_)
        #
        if ids_to_mark is not None:
            for i_grp, id in enumerate(ids_to_mark):
                mask = trn_labels_[reorder] == id
                distances_trn_trn_[mask, :8] = distances_trn_trn_.max() * i_grp / len(ids_to_mark)
                distances_trn_trn_[mask, 8:16] = 1 - distances_trn_trn_.max() * i_grp / len(ids_to_mark)
                distances_trn_trn_[mask, 16:24] = distances_trn_trn_.max() * i_grp / len(ids_to_mark)
                distances_trn_trn_[mask, 24:32] = 1 - distances_trn_trn_.max() * i_grp / len(ids_to_mark)
                distances_trn_trn_[mask][:, mask] = 1.2
                for i1 in np.argwhere(mask).flatten():
                    for i2 in np.argwhere(mask).flatten():
                        distances_trn_trn_[i1, i2] = 1.2
        #
        sns.heatmap(distances_trn_trn_[:3000, :3000], ax=ax1.flatten()[i], square=True)
        display_distances.append(distances_trn_trn_)
        #print_memory()
        contrast, inner, outer = evaluate_contrast(distances_trn_trn_, trn_labels_)
        ax1.flatten()[i].set_title(feat_version_names[i] + ' (S)\n' + str(np.round(contrast, decimals=2)) + '  {:5.3f} / {:5.3f}'.format(inner, outer))
        del trn_labels_, distances_trn_trn_
        gc.collect()
        # _, _, _, _, distances_trn_trn, _, _ = calc_distances(feat_versions[i], labels, metric='euclidian')
        # distances_trn_trn = distances_trn_trn[reorder][:, reorder]
        # sns.heatmap(distances_trn_trn[:1000, :1000], ax=ax2.flatten()[i], square=True)
        # #print_memory()
        # contrast = evaluate_contrast(distances_trn_trn, trn_labels)
        # ax2.flatten()[i].set_title(feat_version_names[i] + ' (E)\n' + str(np.round(contrast, decimals=2)))
        #
        # distance distributions on ax2
        inner_distances, outer_distances = np.zeros(0), np.zeros(0)
        for label in tqdm(np.unique(trn_labels)):
            mask = trn_labels == label
            #print(len(inner_distances), mask.shape, mask.sum())
            inner_distances = np.concatenate((inner_distances, distances_trn_trn[mask][:, mask].flatten()))
            outer_distances = np.concatenate((outer_distances, distances_trn_trn[mask][:, ~mask].flatten()))
        h_inner, x_inner = np.histogram(inner_distances, bins=np.linspace(start=0, stop=1, num=51))
        h_outer, x_outer = np.histogram(outer_distances, bins=np.linspace(start=0, stop=1, num=51))
        # print(inner_distances.size, inner_distances.min(), inner_distances.max())
        # print(x_inner, h_inner)
        # print(outer_distances.size, outer_distances.min(), outer_distances.max())
        # print(x_outer, h_outer)
        del distances_trn_trn, inner_distances, outer_distances
        gc.collect()
        ax2.flatten()[i].plot((x_inner[:-1] + x_inner[1:]) / 2, h_inner / h_inner.sum())
        ax2.flatten()[i].plot((x_outer[:-1] + x_outer[1:]) / 2, h_outer / h_outer.sum())
        ax2.flatten()[i].set_title(feat_version_names[i] + ' (S)\n' + str(np.round(contrast, decimals=2)) + '  {:5.3f} / {:5.3f}'.format(inner, outer))
        ax2.flatten()[i+1].plot((x_inner[:-1] + x_inner[1:]) / 2, np.cumsum(h_inner) / h_inner.sum())
        ax2.flatten()[i+1].plot((x_outer[:-1] + x_outer[1:]) / 2, np.cumsum(h_outer) / h_outer.sum())
        ax2.flatten()[i + 1].grid(True)
        #ax2.flatten()[i].set_title(feat_version_names[i] + ' (S)\n' + str(np.round(contrast, decimals=2)) + '  {:5.3f} / {:5.3f}'.format(inner, outer))
        display_hists.append([[(x_inner[:-1] + x_inner[1:]) / 2, h_inner / h_inner.sum()], [(x_outer[:-1] + x_outer[1:]) / 2, h_outer / h_outer.sum()]])
        #
        feat_versions[i] = None
        gc.collect()
        #print_memory()

    return fig, display_distances, display_hists, trn_label_list


# search for IDs wirh large number of samples
def show_diminant_IDs(dataset):

    counts = np.bincount(dataset.labels[dataset.labels > -1])
    ordered = np.argsort(counts)[::-1]
    [print(id, counts[id]) for id in ordered[:20]]
    fig, ax = plt.subplots(4, 5, figsize=(16, 12))
    [ax_.axis('off') for ax_ in ax.flatten()]
    for col_id, id in enumerate(ordered[:5]):
        ax[0, col_id].set_title('ID {}'.format(id))
        idxs = np.random.choice(np.argwhere(dataset.labels == id).flatten(), 4)
        for row_id, idx in enumerate(idxs):
            img, label = dataset[idx]
            assert label == id
            ax[row_id, col_id].imshow(img)
    plt.show(block=False)

    return ordered


if __name__ == '__main__':

    #ROOT_FOLDER = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026'

    names = ['SalamanderID2025', 'SeaTurtleID2022', 'LynxID2025', 'TexasHornedLizards']
    # model_names = ['Mega-384']#, 'DINOv2', 'sigLip']#['Mega-224', 'Mega-384', 'miewid']
    # extract_modes = ['', '_s', '_enh_rfnd', '_enh_rfnd_s']# '_blr', '_mix']

    feat_files = dict({'SalamanderID2025': os.path.join(ROOT_FEATURES, 'SalamanderID2025_Mega-384.npz'),
                       'SeaTurtleID2022': os.path.join(ROOT_FEATURES, 'mega384_crefined_SeaTurtleID2022_mmr.npz'),
                       #'LynxID2025': os.path.join(ROOT_FEATURES, 'LynxID2025_resnet.cls.npz'),#os.path.join(ROOT_FEATURES, 'LynxID2025_Mega-384_rfnd.npz'),#
                       'LynxID2025': os.path.join(ROOT_MODELS, 'Lynx models', 'model 5', 'LynxID2025_resnet.npz'),
                       'TexasHornedLizards': ''})


    # dataset statistics
    for db_name in names[2:3]:

        try:
            data = np.load(feat_files[db_name])
            labels = data['all_labels']
            print(db_name, (labels > -1).sum(), (labels == -1).sum())
            ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))
        except:
            continue

    # # compare different features
    # for name in names[:3]:
    #     features_list, featues_names = [], []
    #     for model_name in model_names:
    #         for extract_mode in extract_modes:
    #
    #             fname = os.path.join(ROOT_FOLDER, 'features', '{}_{}{}.npz'.format(name, model_name, extract_mode))
    #             data = np.load(fname)
    #             labels = data['all_labels']
    #             featues_names.append(model_name + extract_mode)
    #             features_list.append(data['all_features'].squeeze())
    #     fig, _, _ = visualize_distances(feat_version_names=featues_names, feat_versions=features_list, labels=labels)
    #     fig.suptitle(name)
    #     # analyze_feture_vector(features_list[0], labels)
    # plt.show()


    dataset_full = AnimalCLEF2026(
        ROOT_DATA,
        transform=None,
        load_label=True,  # return label as 2nd parameter
        factorize_label=True,  # replace string for unique integer
        check_files=False
    )
    base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)
    ids_to_mark = show_diminant_IDs(dataset=base_dataset)
    ids_to_mark = []#[1, 26, 10, 21, 25]


    # single feature model per specious
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    #[ax[3, i1].axis('off') for i1 in range(2)]
    #model_names = ['miewid' for name in names]
    for i, db_name in enumerate(names[2:3]):
        fname = feat_files[db_name]
        if fname == '':
            continue

        features_list, featues_names = [], []
        data = np.load(fname)
        features = data['all_features']
        labels = data['all_labels']
        ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))
        featues_names.append(fname)
        features_list.append(data['all_features'].squeeze())

        #
        # ID packing (lynx)
        big_id_list = [1, 26, 5, 8, 10, 23, 21, 14, 27, 40, 9, 25, 4, 20, 16, 6]
        CLASSIFIER = True
        if CLASSIFIER:
            for i1 in range(len(labels)):
                if labels[i1] != -1:
                    if (not labels[i1] in big_id_list):# or (labels[i1] == -1):
                        labels[i1] = 0
                    else:
                        #print(i1, labels[i1])
                        labels[i1] = int(np.argwhere(labels[i1] == big_id_list).squeeze()) + 1
            #
            dtct = np.argmax(features.squeeze(), axis=1)
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(labels, dtct)
            fig_cm, ax_cm = plt.subplots(1, 1)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(labels),
                                    yticklabels=np.unique(labels))
            acc = np.diag(cm[1:][:, 1:]).sum() / cm[1:][:, 1:].sum()
            fig_cm.suptitle('accuracy (on train examples) = {:5.2f}'.format(acc))
            #
            # document detections of test
            test_dtct_list = dict()
            test_idxs = np.argwhere(labels == -1).flatten().astype(int)
            test_dtcts = np.argmax(features.squeeze(), axis=1)[test_idxs]
            for id in np.unique(labels[labels > -1]):
                if id > 0:
                    id_pos = np.argwhere(test_dtcts == id).flatten().astype(int)
                    print('{} ({}) :\t'.format(id, len(id_pos)), id_pos)
                    test_dtct_list[id] = id_pos

        else:
            # remove labels
            for i1 in range(len(labels)):
                if labels[i1] in big_id_list:
                    labels[i1] = -1

        if i < 3:
            _, distances, hists, trn_label = visualize_distances(feat_version_names=featues_names, feat_versions=features_list, labels=labels)
            ax[i, 0].axis('off')
            sns.heatmap(distances[0][:1000, :1000], ax=ax[i, 0], square=True)
            ax[i, 2].set_ylabel(db_name[:-4][:12])
            ax[i, 2].yaxis.set_label_position("right")
            [ax[i, 1].plot(hists[0][i1][0], hists[0][i1][1]) for i1 in range(2)]
            ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))
            ax[i, 2].bar(np.arange(ID_size_hist.size), ID_size_hist)
            #ax[i, 2].legend('{} trn, {} test'.format((labels > -1).sum(), (labels == -1).sum()))

            # bad IDs analysis
            for id in np.unique(trn_label[0]):
                mask_in = trn_label[0] == id
                mask_out = trn_label[0] != id
                max_dist_in = distances[0][mask_in][:, mask_in].max()
                min_dist_out = np.median(distances[0][mask_in][:, mask_out])
                if (min_dist_out < 0.2) and (mask_in.sum() > 50):#(max_dist_in > min_dist_out) or (min_dist_out < 0.2):
                    print(id, mask_in.sum(), max_dist_in, min_dist_out)
    plt.show()
    fig, ax = plt.subplots(4, 5, figsize=(16, 12))
    [ax_.set_axis('off') for ax_ in ax.flatten()]

