from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import sys
from PIL import Image

from threading import Thread

import traceback

thread_ct = 0
MAX_THREADS = 100

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()               

def intensityImg(im):
    
    intensities = im[:,:,0] * 0.30 + im[:,:,1] * 0.59 + im[:,:,2] * 0.11

    #normalize color intensities
    intensities = intensities / np.max(intensities)
    
    return intensities


def processImage(local_file_in, local_file_out, face_coords, bounds_ratio):
    try:
        img = imread(local_file_in)
        print img.shape

        #TODO image_bounds
        real_height = face_coords[3] - face_coords[1]
        new_height = (face_coords[2] - face_coords[0]) * bounds_ratio
        hdiff = int(real_height - new_height)

        img_processed = Image.fromarray(
            img[
                    face_coords[1]:face_coords[3],
                    face_coords[0]+hdiff/2:face_coords[2]-(hdiff/2),
                    :]
            )

        img_thumb = img_processed.resize((227, 227),
            resample=Image.BILINEAR)

        img_thumb.save(local_file_out, "png")
    except Exception as e:
        print("error processing %s -> %s %s" % 
            (local_file_in, local_file_out, face_coords))
        traceback.print_exc(e)
        


def make_actor_dirs(localpath, actor_name):
    print "making actor dirs for %s in %s" % (actor_name, localpath)
    name = actor_name.replace(" ","_")

    dir_unprocessed = os.path.join(localpath, "unprocessed")
    dir_processed = os.path.join(localpath, "processed")
    if not os.path.exists(dir_unprocessed):
        os.mkdir(dir_unprocessed)
    if not os.path.exists(dir_processed):
        os.mkdir(dir_processed)

    actor_dirname = os.path.join(dir_unprocessed, name)
    if not os.path.exists(actor_dirname):
        os.mkdir(actor_dirname)

    actor_dirname = os.path.join(dir_processed, name)
    if not os.path.exists(actor_dirname):
        os.mkdir(actor_dirname)


def doAll(path, localpath):

    seen_actors = set()

    bounds_ratio = 0.0
    smallest_width = -1
    for line in open(path):
        spl = line.split("\t")
        coords = map(lambda a: int(a), spl[4].split(","))

        width = coords[2] - coords[0]
        c_ratio = float(width) / (coords[3] - coords[1])
        if c_ratio > bounds_ratio:
            bounds_ratio = c_ratio
        if smallest_width == -1 or width < smallest_width:
            smallest_width = width

    print "bounds_ratio: %s, width:%spx"%(bounds_ratio, smallest_width)


    for i,line in enumerate(open(path), 1):
        # A version without timeout (uncomment in case you need to 
        # unsupress exceptions, which timeout() does)
        # testfile.retrieve(line.split()[4], "unprocessed/"+filename)
        # timeout is used to stop downloading images which take too long to download

        #  helper variables
        spl = line.split("\t")
        person_name = spl[0]

        if person_name not in seen_actors:
            seen_actors.add(person_name)
            make_actor_dirs(localpath, person_name)

        person_name = person_name.replace(" ","_")
        face_coords = map(lambda a: int(a), spl[4].split(","))
        url = spl[3]
        extension = url.split('.')[-1]

        local_file = os.path.join(
            person_name, str(i) + "." + extension)
        local_file_full = os.path.join(
            localpath, "unprocessed", local_file)

        # print local_file_full

        #load the file with timeout
        timeout(testfile.retrieve, 
            (url, local_file_full), {}, 0.1)

        # on fail, print msg and continue
        if not os.path.isfile(local_file_full):
            print "..fetching file failed <%s>"%(url)

        # otherwise, process the image
        else:
            # print("processing " + local_file)
            # print url, face_coords

            processImage(
                local_file_full,
                os.path.join(localpath, "processed", local_file),
                face_coords, bounds_ratio)


# print "created processed/%s"%(local_file)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "usage: %s <data file> <local path>"
        sys.exit(1)

    doAll(sys.argv[1], sys.argv[2])
