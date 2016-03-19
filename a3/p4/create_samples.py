import os
import sys
import random
import shutil

samplesets = {
    "training": 100,
    "validation": 10,
    "test": 10
}

def clearDir(directory):
    if os.path.exists(directory):
        for path in os.listdir(directory):
            os.remove(os.path.join(directory, path))


def copyToDir(fromPath, destDir):
    if not os.path.exists(destDir):
        os.mkdir(destDir)

    # print os.path.join(
    #         destDir, 
    #         os.path.basename(fromPath))

    shutil.copyfile(fromPath, 
        os.path.join(
            destDir, 
            os.path.basename(fromPath)
        ))

def sampleDirectory(basepath):

    samples = [os.path.join(basepath, sample)
                for sample in os.listdir(basepath)
                if not os.path.isdir(os.path.join(basepath, sample))]

    # create a subsample of the right size
    numvalues = sum(samplesets.values())
    if len(samples) < numvalues:
        print("not enough values in directory %s" % basepath)
        sys.exit(1)

    body = random.sample(samples, numvalues)
    random.shuffle(body)

    # divide up the subsample and copy the files into
    # the right directories
    samples = {}
    for sampleset, count in samplesets.iteritems():
        samples[sampleset] = body[:count]
        body = body[count:]

    for key in samples:
        clearDir(os.path.join(basepath, key))
        for f in samples[key]:
            # print "%s -> %s" % (f, os.path.join(basepath, key))
            copyToDir(f, os.path.join(basepath, key))

def main():
    # get the path we are working off of out of the args
    if len(sys.argv) != 2:
        print("usage: %s <path>" % sys.argv[0])
        sys.exit(1)

    sampleDirectory(sys.argv[1])

if __name__ == "__main__":
    main()
