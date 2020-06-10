from pprint import pprint
from random import shuffle
from math import ceil
import numpy as np


def func(x):
    return x[0] ^ (not x[4])


K = 0.5
N = 6
M = 1 << N
NEED = ceil(M * K)
training = list()
for mask in range(1 << N):
    arr = list(map(int, bin(mask)[2:].rjust(N, '0')))
    training.append([arr, func(arr)])
shuffle(training)

print('training input:')
X = [x[0] for x in training[:NEED]]
pprint(X)
print('training output:')
Y = [x[1] for x in training[:NEED]]
print(Y)
print('input:')
I = [x[0] for x in training[NEED:]]
pprint(I)
print('output:')
O = [x[1] for x in training[NEED:]]
print(O)
