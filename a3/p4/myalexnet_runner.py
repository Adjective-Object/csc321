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

def eval_performance(inset, outset):
    num = 0
    for x in range(inset.shape[0]):
        if argmax(outset) == argmax_prediction.eval(
            session=sess, 
            feed_dict={
                network_input: inset[x, :].reshape(1, inset.shape[1])
            }):
            num += 1
    return float(num) / inset.shape[0]

def train_batch(training_batch, training_outputs):
    for x in range(training_batch.shape[0]):
        i = training_batch[x,:].reshape(1, training_batch.shape[1])
        e = training_outputs[x,:].reshape(1, training_outputs.shape[1])

        sess.run(train_step, feed_dict={
            network_input: i,
            network_expected: e
        })

def eval_batch(all_train, all_test, all_valid):
    tr = eval_performance(all_train[1], all_train[1]),

    te = eval_performance(all_test[1], all_test[1]),

    va = eval_performance(all_valid[1], all_valid[1])

    return tr, te, va


def train(data, passes=100, bsize=1, snapshot_frequency=100):
    all_train = face_utils.load_fileset_multichannel(data["training"], "training",   0, None, 4096)
    all_test  = face_utils.load_fileset_multichannel(data["training"], "test",       0, None, 4096)
    all_valid = face_utils.load_fileset_multichannel(data["training"], "validation", 0, None, 4096)

   # do 100 passes of the training set
    rates = []
    print
    for p in range(passes):
        pcomp = int(float(p)/passes * 30)
        sys.stdout.write("\r pass %d of %d [%s%s] training.. " % (p, passes, "#" * pcomp, " " * (30 - pcomp)))
        sys.stdout.flush()

        i = 0
        while True:
            # load the current batch
            names, training_batch, training_outputs = face_utils.load_fileset_multichannel(
                data["training"],
                "training",
                batch_low=i * bsize,
                batch_high=(i + 1) * bsize,
                dimension=4096)
            
            if len(names) == 0:
                break;

            train_batch(training_batch, training_outputs)

            i += 1

        sys.stdout.write("\r pass %d of %d [%s%s] evaluating.." % (p, passes, "#" * pcomp, " " * (30 - pcomp)))
        sys.stdout.flush()
        rates.append(eval_batch(all_train, all_test, all_valid))

        sys.stdout.write("\r pass %d of %d [%s%s] dumping..   " % (p, passes, "#" * pcomp, " " * (30 - pcomp)))
        sys.stdout.flush()
        dump_snapshot(rates, "_pass_%04d" % (p))

    rates.append(train_batch(all_train, all_test, all_valid, training_batch, training_outputs))
    dump_snapshot(rates, "_FINAL")

    print rates
    te = plt.plot([x[0] for x in rates], label="test")
    tr = plt.plot([x[1] for x in rates], label="train")
    va = plt.plot([x[2] for x in rates], label="validation")
    plt.legend(loc=4)
    show()


def dump_snapshot(rates, i):
    snapshot = {}
    for key in net_data.keys():
        snapshot[key] = [None, None]
        snapshot[key][0] = sess.run(net_data[key][0])
        snapshot[key][1] = sess.run(net_data[key][1])

    snapshot["rates"] = rates
        
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
