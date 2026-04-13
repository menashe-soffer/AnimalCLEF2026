import torch
from torch.utils.data import Sampler
import numpy as np
import random


class PKSampler(Sampler):
    def __init__(self, anchors, labels, p=2, k=4):
        self.p = p
        self.k = k

        # Mapping: ID -> [list of POSITIONS in the anchors list]
        self.id_to_anchor_positions = {}

        # 'pos' goes from 0 to 2364
        # 'idx' is the global image ID (e.g. 5021)
        for pos, (idx, lbl) in enumerate(zip(anchors, labels)):
            lbl = int(lbl)
            if lbl not in self.id_to_anchor_positions:
                self.id_to_anchor_positions[lbl] = []

            # CRITICAL: We store 'pos', the index into the 'anchors' list
            self.id_to_anchor_positions[lbl].append(pos)

        self.unique_ids = list(self.id_to_anchor_positions.keys())

        # Identity Weights (Inverse Sqrt frequency)
        counts = np.array([len(self.id_to_anchor_positions[i]) for i in self.unique_ids])
        self.id_weights = 1.0 / np.sqrt(np.maximum(counts, 3))
        self.id_weights /= self.id_weights.sum()

        self.num_batches = len(anchors) // (p * k)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            selected_ids = np.random.choice(
                self.unique_ids, size=self.p, replace=False, p=self.id_weights
            )

            for identity in selected_ids:
                positions = self.id_to_anchor_positions[identity]
                sel = np.random.choice(positions, size=self.k, replace=(len(positions) < self.k))
                batch.extend(sel.tolist())

            random.shuffle(batch)
            yield batch  # Now yields numbers between 0 and 2364


    def __len__(self):
        return self.num_batches




import torch.nn as nn
import torch.nn.functional as F

class AbsoluteConstraintLoss(nn.Module):
    def __init__(self, pos_margin=0.2, neg_margin=0.5):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, anchor, positive, negative):
        # Calculate Euclidean distances
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)

        # 1. Penalize positives that are too far apart ( > 0.2)
        # We want d_ap to be small.
        loss_pos = torch.mean(F.relu(d_ap - self.pos_margin))

        # 2. Penalize negatives that are too close ( < 0.5)
        # We want d_an to be large.
        loss_neg = torch.mean(F.relu(self.neg_margin - d_an))

        # Total loss is the sum
        return loss_pos + loss_neg