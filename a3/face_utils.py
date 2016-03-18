import numpy as np
import io
import os
import sys
import json

from scipy.spatial.distance import euclidean
from scipy.misc import imread

import islice from itertools


def ilen(iterator):
    return len(list(iterator))

def verify_set(kind, *sets):
    """ verifies that the set given has the correct folder structure
    """
    for s in sets:
        if (not os.path.exists(
            os.path.join(
                kind["corpus"], s)) or 
            not os.path.isdir(
                os.path.join(
                    kind["corpus"], s))):
            return False
    return True


def counts_of(kind, *sets):
    """ counts the number of files in the given subsets
    """
    sets = [os.path.join(kind["corpus"], s) for s in sets]
    counts = [ilen(os.listdir(s)) for s in sets]
    return sum(counts)

def get_sample_files(dataset, subset, batch_low, batch_high):
    """ Generates a list of data files from the given subset
    """
    for class_index, d in enumerate(dataset):
        subsetDir = os.path.join(d["corpus"], subset)
        for path in islice(os.listdir(subsetDir), batch_low, batch_high):
            yield os.path.join(subsetDir, path), class_index


def load_fileset(dataset, subset, batch_low, batch_high):

    input_vector = np.ndarray(
        shape=(32 * 32, batch_high - batch_low),
        dtype=float)
    output_vector = np.zeros(len(dataset), batch_high - batch_low);

    samplefiles = list(get_sample_files(dataset, subset), batch_low, batch_high)

    for i, (imgpath, class_index) in enumerate(samplefiles):
        img = imread(imgpath, flatten=True).flatten()
        input_vector[:, i] = img
        output_vector[class_index, i] = 1;

    return samplefiles, data_vector, answer_vector


