import numpy as np

class ID_manager():

    def __init__(self, num_subsets=2, min_for_trn=2):

        self.num_subsets = num_subsets
        self.min_for_trn = min_for_trn


    def load_labels(self, labels):

        self.labels = np.copy(labels)
        self.dset_test_mask = labels == -1
        self.bins = np.bincount(labels[~ self.dset_test_mask])
        #
        self.singltone_group_mask = self.bins == 1
        self.small_group_mask = (~self.singltone_group_mask) * (self.bins < self.min_for_trn)
        self.trn_group_mask = self.bins >= self.min_for_trn
        #
        self.singltone_IDs = np.argwhere(self.singltone_group_mask).flatten().astype(int)
        self.small_group_IDs = np.argwhere(self.small_group_mask).flatten().astype(int)
        self.trn_group_IDs = np.argwhere(self.trn_group_mask).flatten().astype(int)




    def get_subset_for_analysis(self, subset=0):

        assert subset < self.num_subsets
        subset_IDs = self.trn_group_IDs[subset::self.num_subsets]
        exclude_IDs = np.sort(np.concatenate((self.singltone_IDs, self.small_group_IDs)))
        ctrl_IDs = list(set(self.trn_group_IDs) - set(subset_IDs))
        #
        subset_reIDs = (-1) * np.ones_like(self.labels, dtype=int)
        ctrl_reIDs = (-1) * np.ones_like(self.labels, dtype=int)
        subset_mask = np.zeros_like(self.labels, dtype=bool)
        ctrl_mask = np.zeros_like(self.labels, dtype=bool)
        for idx, l in enumerate(self.labels):
            if l > -1:
                subset_mask[idx] = l in subset_IDs
                ctrl_mask[idx] = l in ctrl_IDs
                subset_reIDs[idx] = np.argwhere(l == subset_IDs).squeeze() if subset_mask[idx] else -1
                ctrl_reIDs[idx] = np.argwhere(l == ctrl_IDs).squeeze() if ctrl_mask[idx] else -1
            else:
                pass # no re-label, remain out of everything

        return subset_IDs, ctrl_IDs, subset_mask, ctrl_mask, subset_reIDs, ctrl_reIDs




    def get_subset_for_train(self, subset=0):

        assert subset < self.num_subsets
        subset_IDs = self.trn_group_IDs[subset::self.num_subsets]
        other_subsets_IDs =  list(set(self.trn_group_IDs) - set(subset_IDs))
        exclude_IDs = np.sort(np.concatenate((self.singltone_IDs, self.small_group_IDs, other_subsets_IDs)))

        return subset_IDs, exclude_IDs



