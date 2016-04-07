import numpy as np
import random

np.random.seed(555)
random.seed(555)


def preseed(seed_ixs):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    h = np.zeros(bh.shape)

    for seed_ix in seed_ixs:
        # generate input vector
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1

        # forward pass of the neural net
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
    return h

def sample(h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        # forward pass of the neural net
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # generate input to next time step
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        # record this index
        ixes.append(ix)
    return ixes

if __name__ == "__main__":
    # load the file and unpak the weights
    snapshot = np.load(open("dataset/char-rnn-snapshot.npz"))
    Wxh = snapshot["Wxh"] 
    Whh = snapshot["Whh"]
    Why = snapshot["Why"]
    bh = snapshot["bh"]
    by = snapshot["by"]

    mWxh, mWhh, mWhy = snapshot["mWxh"], snapshot["mWhh"], snapshot["mWhy"]
    mbh, mby = snapshot["mbh"], snapshot["mby"]

    # get other parameters of the matrix
    chars, data_size, vocab_size, char_to_ix, ix_to_char = [
        snapshot[x].tolist() for x in [
            "chars", "data_size", "vocab_size", 
            "char_to_ix", "ix_to_char"
        ]]

    def str2ixs(toConvert):
        return [char_to_ix[c] for c in toConvert]


    for temperature in [1.0]:
        print "######################"
        print "# TEMPERATUE = %0.5f #" % temperature
        print "######################"

        for sample_no in range(10):
            print "## BEGIN SAMPLE %02d" % sample_no
            
            seedString = str2ixs("SEEDSTRING")
            memory_state = preseed(seedString[:-1])

            hallucination = seedString+ sample(memory_state, seedString[-1], 300)
            
            print "".join([ix_to_char[index] for index in hallucination])
            


