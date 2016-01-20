import numpy as np
import os
import sys

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

def get_sample_files(dataset, subset):
    """ Generates a list of data files from the given subset
    """
    for d in dataset:
        subsetDir = os.path.join(d["corpus"], subset)
        for path in os.listdir(subsetDir):
            yield os.path.join(subsetDir, path)


def load_fileset(dataset, subset):

    counts = [counts_of(kind, subset) for kind in dataset]
    
    data_vector = np.ndarray(
        shape=(sum(counts), 32 * 32),
        dtype=float)

    samplefiles = list(get_sample_files(dataset, subset))

    for i, imgpath in enumerate(samplefiles):
        img = imread(imgpath, flatten=True).flatten()
        data_vector[i , :] = img

    return samplefiles, counts, data_vector

def knn(dataSet, testSet):
    """ given a [a x n] dataSet and a [b x n] testSet,
        returns a [b x 1] array of the closest indecies of
        the columns of testSet to columns in testSet;
    """

    output_indices = np.ndarray(
        shape=(testSet.shape[0]),
        dtype=float)

    distances = np.ndarray(
        shape=(dataSet.shape[0]),
        dtype=float)

    for testIndex in range(testSet.shape[0]):
        for dataIndex in range(dataSet.shape[0]):
            distances[dataIndex] = euclidean(
                dataSet[dataIndex, :],
                testSet[testIndex, :])

        # copy the minimum distane index into testIndex
        
        output_indices[testIndex] = np.argmin(
            distances,
            axis=0)

    return output_indices

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

    training_filenames, training_counts, training_set = (
        load_fileset(dataset, "training"))

    def map_training_index_to_class(input_index):
        for i, count in enumerate(training_counts):
            input_index -= count
            if input_index <= 0:
                return dataset[i]
        print input_index, training_counts
        return None

    # get the number of testing files and then put them all
    # in the same numpy array

    testing_filenames, testing_counts, testing_set = (
        load_fileset(dataset, "test"))

    def map_testing_index_to_class(input_index):
        for i, count in enumerate(testing_counts):
            input_index -= count
            if input_index <= 0:
                return dataset[i]
        print input_index, testing_counts
        return None

    # perform the knn junk
    closests = knn(training_set, testing_set)

    # print the output into a nice csv format
    # format:   real filename, real name, real gender,                  \ 
    #           recognized filename, recognized name, recognized gender \
    #           match on name, match on gender                          \

    for input_index, recognized_index in enumerate(closests):
        recognized_class = map_training_index_to_class(recognized_index)
        input_class      = map_testing_index_to_class(input_index)

        if recognized_class == None or input_class == None:
            print("error mapping indcies back to values")
            print("    recognized_index = %s" % recognized_index)
            print("    input_index  = %s" % input_index)
            sys.exit(1)

        print(", ".join([
            training_filenames[int(recognized_index)],
            recognized_class["name"],
            recognized_class["gender"],
            testing_filenames[int(input_index)],
            input_class["name"],
            input_class["gender"],
            "1" if input_class["name"] == recognized_class["name"] else "0",
            "1" if input_class["gender"] == recognized_class["gender"] else "0",
            ]))




if __name__ == "__main__":
    process_dataset(data)
