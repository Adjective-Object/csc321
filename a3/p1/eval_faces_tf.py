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

# eval the faces network
from faces_network import *

def eval_acc(dataset, pickle):
    names, test_inputs, test_outputs = face_utils.load_fileset(
        dataset["training"], "test", 0, None);

    print "accuracy =", accuracy.eval(
        session=sess,
            feed_dict={
                inputs: test_inputs,
                expectation: test_outputs
            })



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "usage: %s <data_json> <snapshot_file>" % sys.argv[0]
        sys.exit(1)

    jsonbody = io.open(sys.argv[1], encoding='utf-8-sig')
    data = json.loads(jsonbody.read())

    with open(sys.argv[2]) as f:
        eval_acc(data, f)

