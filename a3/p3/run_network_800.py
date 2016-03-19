# helpers for loading face shit
import face_utils

import io
import json


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

import os


import pickle

random.seed(555)

from network_800 import *


def do_train(data, passes=100, bsize=10, snapshot_frequency=50):
    all_train = face_utils.load_fileset(data["training"], "training", 0, None)
    all_test  = face_utils.load_fileset(data["training"], "test", 0, None)
    all_valid = face_utils.load_fileset(data["training"], "validation", 0, None)

    # do 100 passes of the training set
    for p in range(passes):
        print "pass (%3s/%3s)" %(p, passes)

        i = 0
        while True:
            # load the current batch
            names, training_batch, training_outputs = face_utils.load_fileset(
                data["training"],
                "training",
                i * bsize, (i + 1) * bsize)

            if len(names) == 0:
                break;

            sess.run(train_step, feed_dict={
                inputs: training_batch,
                expectation: training_outputs
            })
            i += 1

    dump_snapshot("_800_FINAL")

def evaluate_accuracies(i, o):
    return accuracy.eval(
        session=sess,
            feed_dict={
                inputs: i,
                expectation: o 
    })


def dump_snapshot(i):
    print "dumping snapshot for generation %s" % i

    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)
    pickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <data_json>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    do_train(data);

