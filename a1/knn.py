import numpy as np
import os

from scipy.spatial.distance import euclidean
from scipy.misc import imread

data = [
    {"name": "Gerard Butler",
     "gender": "male",
     "corpus": "actors/processed/Gerard_Butler"},
    {"name": "Daniel Radcliffe",
     "gender": "male",
     "corpus": "actors/processed/Daniel_Radcliffe"},
    {"name": "Michael Vartan",
     "gender": "male",
     "corpus": "actors/processed/Michael_Vartan"},

    {"name": "Angie Harmon",
     "gender": "female",
     "corpus": "actresses/processed/Angie_Harmon"},
    {"name": "Lorraine Bracco",
     "gender": "female",
     "corpus": "actresses/processed/Lorraine_Bracco"},
    {"name": "Peri Gilpin",
     "gender": "female",
     "corpus": "actresses/processed/Peri_Gilpin"},
]

#########################################
# Helpers for dealing with the expected #
# structure of the filesystem           #
#########################################

def ilen(iterator):
    ct = 0
    for i in iterator:
        ct += 1
    return ct

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
    return sum(
        [ilen(os.listdir(
            os.path.join(kind["corpus"], s))
        for s in sets)])

def get_sample_files(dataset, subset):
    """ Generates a list of data files from the given subset
    """
    for d in dataset:
        for path in os.path.join(d["corpus"], subset):
            yield os.path.join(d["corpus"], subset, path)


def load_fileset(dataset, subset):

    counts = [counts_of(kind, subset) for kind in dataset]

    data_vector = np.ndarray(
        shape=(sum(counts), 32 * 32),
        dtype=float)

    enumeratedFiles = enumerate(
        get_sample_files(dataset, subset))

    for i, imgpath in enumeratedFiles:
        data_vector[i , :] = imread(imgpath).flatten()

    return counts, data_vector

def knn(dataSet, testSet):
    """ given a [a x n] dataSet and a [b x n] testSet,
        returns a [b x 1] array of the closest indecies of
        the columns of testSet to columns in testSet;
    """

    output_indices = np.ndarray(
        shape=(testSet.shape[0], 1),
        dtype=float)

    distances = np.ndarray(
        shape=(dataSet.shape[0], 1),
        dtype=float)

    for testIndex in testSet.shape[0]:
        for dataIndex in dataSet.shape[0]:
            distances[dataIndex, 0] = euclidean(
                dataSet[dataIndex, :],
                testSet[testIndex, :])

        # copy the minimum distane index into testIndex
        
        output[testIndex] = np.argmin(
            distances,
            axis=0)

    return output

def process_dataset(dataset):
    # check that all the kinds have the right subfolders
    for kind in dataset:
        if not verify_set(
                kind,
                "training",
                "test",
                "validation"):
            print("%s is missing some dataset.")
            sys.exit(1)

    # get the number of training files and then put them all
    # in the same numpy array, create a function mapping
    # an index in training to a class

    training_counts, training_set = load_fileset(dataset, "training")
    def map_training_index_to_class(input_index):
        for i, count in enumerated(training_counts):
            input_index -= count
            if input_index <= 0:
                return dataset[i]
        return None

    # get the number of testing files and then put them all
    # in the same numpy array

    testing_counts, testing_set = load_fileset(dataset, "test")
    def map_testing_index_to_class(input_index):
        for i, count in enumerated(testing_counts):
            input_index -= count
            if input_index <= 0:
                return i
        return -1

    closests = knn(training_set, testing_set)
    for testing_index, training_index in enumerated(closests):
        print("training index = %s, testing_index = %s" % 
                (i, closest_index))
        print("training_class =")
        print map_testing_index_to_class()
        print map_testing_index_to_class()





if __name__ == "__main__":
    process_dataset(data)
