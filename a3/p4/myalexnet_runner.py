from numpy import *
import os, sys, io, json, face_utils
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import tensorflow as tf

from myalexnet import *

def train(data, passes=100, bsize=10, snapshot_frequency=50):
    all_train = face_utils.load_fileset_multichannel(data["training"], "training",   0, None, 4096)
    all_test  = face_utils.load_fileset_multichannel(data["training"], "test",       0, None, 4096)
    all_valid = face_utils.load_fileset_multichannel(data["training"], "validation", 0, None, 4096)

    rate = []

    # do 100 passes of the training set
    for passes in range(passes):
        i = 0
        while True:
            # load the current batch
            names, training_batch, training_outputs = face_utils.load_fileset_multichannel(
                data["training"],
                "training",
                batch_low=i * bsize,
                batch_high=(i + 1) * bsize,
                dimension=4096)
            print training_batch.shape, training_outputs.shape

            if len(names) == 0:
                break;

            sess.run(train_step, feed_dict={
                network_input: training_batch,
                network_expected: training_outputs
            })

            # evaluate the accuracies on the test, training, and validation sets
            accs = (
                evaluate_accuracies(all_train[1], all_train[2]),
                evaluate_accuracies(all_test[1], all_test[2]),
                evaluate_accuracies(all_valid[1], all_valid[2]),
            ) 

            print "test: %4f train: %4f valid: %4f" % accs
            rate.append(accs)

            if i % snapshot_frequency == 0:
                dump_snapshot("%04d_%04d" % (passes, i))

            i += 1

    dump_snapshot("FINAL")

    print rate
    te = plt.plot([r[0] for r in rate], label="test")
    tr = plt.plot([r[1] for r in rate], label="train")
    va = plt.plot([r[2] for r in rate], label="validation")
    plt.legend(loc=4)
    show()


def evaluate_accuracies(i, o):
    return accuracy.eval(session=sess, feed_dict={
            network_input: i, network_expected: o})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <data_json>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    train(data);
