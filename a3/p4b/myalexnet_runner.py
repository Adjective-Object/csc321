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

current_generation = 0
rates = []
from myalexnet import *

def eval_performance(inset, outset):
    num = 0.0
    for i in range(inset.shape[0]):
        num += sess.run(accuracy, feed_dict={
            x: inset[i,:].reshape(1, 277, 277, 3),
            y: outset[i,:].reshape(1, outset.shape[1])
        })
    return float(num) / float(inset.shape[0])

def train_batch(training_batch, training_outputs):
    for j in range(training_batch.shape[0]):
        i = training_batch[j,:].reshape(277, 277, 3),
        e = training_outputs[j,:].reshape(1, training_outputs.shape[1])

        sess.run(train_step, feed_dict={
            x: i,
            y: e
        })

def eval_batch(all_train, all_test, all_valid):
    tr = eval_performance(all_train[1], all_train[2])
    te = eval_performance(all_test[1], all_test[2])
    va = eval_performance(all_valid[1], all_valid[2])

    return (tr, te, va)


def train(data, passes=100, bsize=1, snapshot_frequency=100):
    global rates

    print "loading images"

    all_train = face_utils.load_fileset_multichannel(data["training"], "training",   0, 10, 277 ** 2)
    all_test  = face_utils.load_fileset_multichannel(data["training"], "test",       0, None, 277 ** 2)
    all_valid = face_utils.load_fileset_multichannel(data["training"], "validation", 0, None, 277 ** 2)

    # do 100 passes of the training set
    print
    for p in range(current_generation, passes):
        pcomp = int(float(p)/passes * 30)
        sys.stdout.write("\r pass %d of %d [%s%s] training.. " % (p, passes, "#" * pcomp, " " * (30 - pcomp)))
        sys.stdout.flush()

        train_batch(all_train[1], all_train[2])

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
    snapshot["outW"] = sess.run(outW)
    snapshot["outb"] = sess.run(outb)
    snapshot["rates"] = rates

    pickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))



def evaluate_accuracies(i, o):
    return accuracy.eval(session=sess, feed_dict={
            x: i, y: o})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: %s <data_json>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    train(data);
