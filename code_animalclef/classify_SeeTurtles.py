import numpy as np
from class_utils import *


def classify_SeeTurtle(features, known_labels):

    _, _, _, _, _, _, _, distances = calc_distances(features=features, labels=known_labels)
    labels = classify_using_knowns(distances=distances, labels=known_labels)

    return labels