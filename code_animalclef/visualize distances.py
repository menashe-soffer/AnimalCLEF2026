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

from class_utils import calc_distances

def print_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Convert bytes to Gigabytes
    gb_usage = mem_info.rss / (1024 ** 3)
    print(f"Current RAM Usage: {gb_usage:.2f} GB")




def visualize_distances(feat_version_names, feat_versions, labels):

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

    fig, ax = plt.subplots(plot_rows, 2 * plot_cols, figsize=(16, 12), sharex=False, sharey=False)
    ax1 = ax[:, :plot_cols] if plot_rows > 1 else ax[:plot_cols]
    ax2 = ax[:, plot_cols:] if plot_rows > 1 else ax[plot_cols:]

    display_distances, display_hists = [], []
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
        sns.heatmap(distances_trn_trn_[:1000, :1000], ax=ax1.flatten()[i], square=True)
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
        display_hists.append([[(x_inner[:-1] + x_inner[1:]) / 2, h_inner / h_inner.sum()], [(x_outer[:-1] + x_outer[1:]) / 2, h_outer / h_outer.sum()]])
        #
        feat_versions[i] = None
        gc.collect()
        #print_memory()

    return fig, display_distances, display_hists





if __name__ == '__main__':

    ROOT_FOLDER = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026'

    names = ['SeaTurtleID2022', 'SalamanderID2025', 'LynxID2025', 'TexasHornedLizards']
    model_names = ['Mega-384']#, 'DINOv2', 'sigLip']#['Mega-224', 'Mega-384', 'miewid']
    extract_modes = ['', '_s', '_enh_rfnd', '_enh_rfnd_s']# '_blr', '_mix']

    # dataset statistics
    for name in names:
        fname = os.path.join(ROOT_FOLDER, 'features', '{}_{}{}.npz'.format(name, 'Mega-384', ''))
        data = np.load(fname)
        labels = data['all_labels']
        print(name, (labels > -1).sum(), (labels == -1).sum())
        ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))

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


    # single feature model per specious
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    #[ax[3, i1].axis('off') for i1 in range(2)]
    model_names = ['miewid' for name in names]
    for i, (db_name, model_name) in enumerate(zip(names, model_names)):
        features_list, featues_names = [], []
        fname = os.path.join(ROOT_FOLDER, 'features', '{}_{}.npz'.format(db_name, model_name))
        data = np.load(fname)
        features = data['all_features']
        labels = data['all_labels']
        ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))
        featues_names.append(model_name)
        features_list.append(data['all_features'].squeeze())

        if i < 3:
            _, distances, hists = visualize_distances(feat_version_names=featues_names, feat_versions=features_list, labels=labels)
            ax[i, 0].axis('off')
            sns.heatmap(distances[0][:1000, :1000], ax=ax[i, 0], square=True)
            ax[i, 2].set_ylabel(db_name[:-4][:12])
            ax[i, 2].yaxis.set_label_position("right")
            [ax[i, 1].plot(hists[0][i1][0], hists[0][i1][1]) for i1 in range(2)]
            ID_size_hist = np.bincount(np.bincount(labels[labels > -1]))
            ax[i, 2].bar(np.arange(ID_size_hist.size), ID_size_hist)
            #ax[i, 2].legend('{} trn, {} test'.format((labels > -1).sum(), (labels == -1).sum()))
    plt.show()


