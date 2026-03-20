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
            inside_dist = distances[mask][:, mask]
            inner_sum += inside_dist.sum()
            inner_cnt += inside_dist.size
            outer_dist = distances[mask][:, ~mask]
            outer_sum += outer_dist.sum()
            outer_cnt += outer_dist.size

        inner_avg = inner_sum / inner_cnt
        outer_avg = outer_sum / outer_cnt

        return 1 - inner_avg / outer_avg, inner_avg, outer_avg


    num_versions = len(feat_versions)
    plot_rows = int(np.ceil(np.sqrt(num_versions)))
    plot_cols = int(np.ceil(num_versions / plot_rows))

    fig, ax = plt.subplots(plot_rows, 2 * plot_cols, figsize=(16, 12), sharex=True, sharey=True)
    ax1 = ax[:, :plot_cols]
    ax2 = ax[:, plot_cols:]
    for i in range(num_versions):
        #print_memory()
        trn_labels, _, _, _, distances_trn_trn, _, _, _ = calc_distances(feat_versions[i], labels, metric='similarity')
        good_labels = np.argwhere(np.bincount(trn_labels) > 6).flatten()
        mask = np.array([l in good_labels for l in trn_labels])
        trn_labels = trn_labels[mask]
        distances_trn_trn = distances_trn_trn[mask][:, mask]
        #print_memory()
        reorder = np.argsort(trn_labels)
        distances_trn_trn = distances_trn_trn[reorder][:, reorder]
        sns.heatmap(distances_trn_trn[:1000, :1000], ax=ax1.flatten()[i], square=True)
        #print_memory()
        contrast, inner, outer = evaluate_contrast(distances_trn_trn, trn_labels)
        ax1.flatten()[i].set_title(feat_version_names[i] + ' (S)\n' + str(np.round(contrast, decimals=2)) + '  {:5.3f} / {:5.3f}'.format(inner, outer))
        # _, _, _, _, distances_trn_trn, _, _ = calc_distances(feat_versions[i], labels, metric='euclidian')
        # distances_trn_trn = distances_trn_trn[reorder][:, reorder]
        # sns.heatmap(distances_trn_trn[:1000, :1000], ax=ax2.flatten()[i], square=True)
        # #print_memory()
        # contrast = evaluate_contrast(distances_trn_trn, trn_labels)
        # ax2.flatten()[i].set_title(feat_version_names[i] + ' (E)\n' + str(np.round(contrast, decimals=2)))
        feat_versions[i] = None
        gc.collect()
        #print_memory()

    return fig


if __name__ == '__main__':

    ROOT_FOLDER = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026'

    names = ['SeaTurtleID2022', 'SalamanderID2025', 'LynxID2025', 'TexasHornedLizards'][:3]
    model_names = ['Mega-384', 'DINOv2', 'sigLip']#['Mega-224', 'Mega-384', 'miewid']
    extract_modes = ['', '_s']# '_blr', '_mix']

    for name in names:
        features_list, featues_names = [], []
        for model_name in model_names:
            for extract_mode in extract_modes:

                fname = os.path.join(ROOT_FOLDER, 'features', '{}_{}{}.npz'.format(name, model_name, extract_mode))
                data = np.load(fname)
                labels = data['all_labels']
                featues_names.append(model_name + extract_mode)
                features_list.append(data['all_features'].squeeze())
        fig = visualize_distances(feat_version_names=featues_names, feat_versions=features_list, labels=labels)
        fig.suptitle(name)
    plt.show()


