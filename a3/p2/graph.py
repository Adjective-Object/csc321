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
import pickle

current_generation = 0

x = pickle.load(open("new_snapshot_pass_0099.pkl"))

rates = x["rates"]


print rates
te = plt.plot([x[0] for x in rates], label="test")
tr = plt.plot([x[1] for x in rates], label="train")
va = plt.plot([x[2] for x in rates], label="validation")
plt.legend(loc=4)
plt.savefig("graph.png")

