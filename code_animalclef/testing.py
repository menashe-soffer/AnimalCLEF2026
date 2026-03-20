import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wildlife_datasets.datasets import AnimalCLEF2026
import os
import glob
import sklearn.metrics

from classify_SeeTurtles import classify_SeeTurtle


if __name__ == '__main__':

    # switches
    sw_make_val_set = True
    sw_use_only_trn = True
    sw_show_only_ids_with_mistakes = True
    sw_save_images_to_files = True
    db_idx = 1

    root = '/media/soffer/TOSHIBA EXT/AnimalCLEF2026/data'
    np.random.seed(1)

    dataset_full = AnimalCLEF2026(root, transform=None, load_label=True, factorize_label=True, check_files=False)
    db_name_list = ['SalamanderID2025', 'SeaTurtleID2022', 'LynxID2025', 'TexasHornedLizards']
    feature_model_dict = {'SalamanderID2025': 'Mega-384',
                             'SeaTurtleID2022': 'Mega-384',
                             'LynxID2025': 'Mega-384',
                             'TexasHornedLizards': 'Mega-384'}
    db_name = db_name_list[db_idx]
    dataset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)
    feat_fname = os.path.join(root.replace('data', 'features'), '{}_{}.npz'.format(db_name, feature_model_dict[db_name]))
    data = np.load(feat_fname)
    features, all_labels = data['all_features'], data['all_labels']
    if sw_use_only_trn:
        subset_mask = dataset.df['split'] == 'train'
        dataset = dataset.get_subset(subset_mask)
        features = features[subset_mask].squeeze()
        all_labels = all_labels[subset_mask]

    labels = np.copy(all_labels)
    TRN_mask = np.array(dataset.df['split'] == 'train')
    VAL_mask = np.zeros_like(TRN_mask, dtype=bool)
    TST_mask = np.array(dataset.df['split'] == 'test')
    if sw_make_val_set:
        all_trn_idxs = np.argwhere(TRN_mask).flatten()
        val_idxs = np.random.choice(all_trn_idxs, size=int(dataset.__len__() * 0.2), replace=False)
        TRN_mask[val_idxs] = False
        VAL_mask[val_idxs] = True
        labels[VAL_mask] = -1



    db_len = dataset.__len__()
    assert db_len == labels.size
    assert db_len == features.shape[0]

    print(dataset.df.head())
    # fig, ax = plt.subplots(4, 4)
    # for idx in range(16):
    #     img, label = dataset.__getitem__(idx)
    #     ax.flatten()[idx].imshow(img)
    #     ax.flatten()[idx].set_title('iD=' + str(label))
    # fig.suptitle('examples from ' + db_name_list[db_idx])
    # plt.show()

    if db_name == 'SalamanderID2025':
        pass

    if db_name == 'SeaTurtleID2022':

        pred_labels = classify_SeeTurtle(features=features, known_labels=labels)

    if db_name == 'LynxID2025':
        pass

    if db_name == 'TexasHornedLizards':
        pass



    if sw_save_images_to_files:
        save_folder = root.replace('data', 'debug')
        os.makedirs(save_folder, exist_ok=True)
        try:
            [os.remove(f) for f in glob.glob(save_folder + '/*.*')]
        except:
            print('could not remove all files in ', save_folder)

    all_ids = np.unique(np.concatenate((labels, pred_labels)))
    for id in all_ids[all_ids > -1]:
        true_set = id == all_labels
        pred_set = id == pred_labels
        pred_miss = true_set * (~pred_set)
        pred_false = (~true_set) * pred_set
        val_set = pred_set * VAL_mask
        tst_set = pred_set * TST_mask

        all_set = true_set | pred_miss | pred_false | val_set | tst_set
        print(id, true_set.sum(), pred_miss.sum(), pred_false.sum(), val_set.sum(), tst_set.sum())
        if (not sw_show_only_ids_with_mistakes) or (pred_miss.sum() + pred_false.sum()+ tst_set.sum() > 0):
            fig, ax = plt.subplots(4, 6, figsize=(16, 12))
            fig.suptitle('ID = ' + str(id))
            if all_set.sum() > 24:
                all_set_ = pred_miss | pred_false | val_set | tst_set
                remaining = 24 - all_set_.sum()
                #print(remaining, true_set.size)
                if remaining > 0:
                    remaining_idxs = np.random.choice(np.argwhere(true_set).flatten(), size=remaining, replace=False)
                    all_set_[remaining_idxs] = True
                if remaining < 0:
                    choose_idxs = np.random.choice(np.argwhere(all_set_).flatten(), size=24, replace=False)
                    all_set_[:] = False
                    all_set_[choose_idxs] = True
            else:
                all_set_ = all_set

            [ax_.axis('off') for ax_ in ax.flatten()]
            for ax_idx, idx  in enumerate(np.argwhere(all_set_).flatten()):
                img, _ = dataset.__getitem__(idx)
                ax_ = ax.flatten()[ax_idx]
                ax_.imshow(img)
                if val_set[idx]:
                    ax_.set_title('VAL')
                if tst_set[idx]:
                    ax_.set_title('TST')
                if pred_miss[idx]:
                    ax_.set_title('miss as ' + str(pred_labels[idx]))
                if pred_false[idx]:
                    ax_.set_title('false; true is ' + str(all_labels[idx]))

            if sw_save_images_to_files:
                save_path = os.path.join(save_folder, str(id) + '.jpg')
                fig.savefig(save_path, format="jpg", dpi=150)
            else:
                plt.show()


            # CRITICAL: This is the only way to truly free the RAM in a loop
            plt.close(fig)

    # TBD deal with outlayers (-1)

    if sw_make_val_set:
        val_cm = sklearn.metrics.confusion_matrix(all_labels[VAL_mask], pred_labels[VAL_mask])
        val_accuracy = np.diag(val_cm).sum() / val_cm.sum()
        print('val accuracy:', val_accuracy)






