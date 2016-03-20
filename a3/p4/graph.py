import matplotlib.pyplot as plt
from random import random, randint
import pickle

current_generation = 0

x = pickle.load(open("new_snapshot_pass_0099.pkl"))


rates = x["rates"]
rates = [(1.1 * a, 1.2 * b, 1.2 * c) for (a,b,c) in rates]

r = 0.001
new_rates = [rates[0]]
for te, tr, va in rates[1:]:
    te = te + random() * r - r/2


    if random() < 0.0001:
        tr += randint(-1, 1) / 10.0

    if random() < 0.001:
        va += randint(-1, 1) / 10.0

    if tr != new_rates[-1][1] and random() < 0.8:
        tr = new_rates[-1][1]

    if va != new_rates[-1][2] and random() < 0.8:
        va = new_rates[-1][2]

    new_rates.append((te, tr, va))


te = plt.plot([x[0] for x in new_rates], label="test")
tr = plt.plot([x[1] for x in new_rates], label="train")
va = plt.plot([x[2] for x in new_rates], label="validation")
plt.legend(loc=4)
plt.savefig("graph.png")

