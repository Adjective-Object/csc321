import numpy as np
import random

np.random.seed(555)
random.seed(555)

def sample(h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
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

    for temperature in (x / 100.0 for x in range(0,100)):
        print "######################"
        print "# TEMPERATUE = %0.5f #" % temperature
        print "######################"

        for sample_no in range(10):
            print "## BEGIN SAMPLE %02d" % sample_no
            # get first char as weighted random choice
            initial_char = random.choice(chars)

            #initial memstate is null
            memory_state = np.zeros(bh.shape)

            hallucination = sample(memory_state, char_to_ix[initial_char], 300)

            print "".join([ix_to_char[index] for index in hallucination])

