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

import pickle
from myalexnet import *

def train_batch(training_batch, training_outputs):
    for x in range(training_batch.shape[0]):
        sess.run(train_step, feed_dict={
            network_input: training_batch[x,:].reshape(1, training_batch.shape[1]),
            network_expected: training_outputs[x,:].reshape(1, training_outputs.shape[1])
        })
   

def train(data, passes=100, bsize=10, snapshot_frequency=100):
    all_train = face_utils.load_fileset_multichannel(data["training"], "training",   0, None, 4096)
    all_test  = face_utils.load_fileset_multichannel(data["training"], "test",       0, None, 4096)
    all_valid = face_utils.load_fileset_multichannel(data["training"], "validation", 0, None, 4096)
    # do 100 passes of the training set
    for p in range(passes):
        print "pass %4d of %4d" %(p, passes)

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

            train_batch(training_batch, training_outputs)

            if i % snapshot_frequency == 0:
                dump_snapshot("%04d_%04d" % (i, passes))

            i += 1

    dump_snapshot("FINAL")


def dump_snapshot(i):
    print "dumping snapshot for generation %s" % i

    snapshot = {}
    for key in net_data.keys():
        snapshot[key] = [None, None]
        snapshot[key][0] = sess.run(net_data[key][0])
        snapshot[key][1] = sess.run(net_data[key][1])
        
    pickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))



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
