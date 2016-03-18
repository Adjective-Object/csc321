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


from faces_network import *


def do_train(data, bsize=4, snapshot_frequency=30):
    for i in range(100):
        # load the current batch
        names, training_batch, training_outputs = face_utils.load_fileset(
            data["training"],
            "training",
            i * bsize, (i + 1) * bsize)

        if len(names) == 0:
            # out of pictures
            print "out of samples in generation %s" % i
            break;


        sess.run(train_step, feed_dict={
            inputs: training_batch,
            expectation: training_outputs
        })

        if i % snapshot_frequency == 0:

            print "accuracy =", accuracy.eval(
                session=sess,
                    feed_dict={
                inputs: training_batch,
                expectation: training_outputs   
            })

            dump_snapshot(i)

    dump_snapshot("FINAL")

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

