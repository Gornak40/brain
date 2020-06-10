import numpy as np
from pprint import pprint
from generator import X, Y, I, O


def activation(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def learn():
    ERAS = 60000
    N = 15
    tInput = np.array(X)
    sInput = len(tInput[0])
    tOutput = np.array([Y]).T
    sOutput = len(tOutput[0])
    syn0 = 2 * np.random.rand(sInput, N) - 1
    syn1 = 2 * np.random.rand(N, sOutput) - 1
    for era in range(ERAS):
        l0 = tInput
        l1 = activation(np.dot(l0, syn0))
        l2 = activation(np.dot(l1, syn1))
        
        l2Error = tOutput - l2
        if not era % 10000:
            print('error: {}'.format(np.mean(np.abs(l2Error))))
        l2Delta = l2Error * activation(l2, deriv=True)
        l1Error = l2Delta.dot(syn1.T)
        l1Delta = l1Error * activation(l1, deriv=True)
        
        syn1 += l1.T.dot(l2Delta)
        syn0 += l0.T.dot(l1Delta)
    
    l0 = np.array(I)
    l1 = activation(np.dot(l0, syn0))
    l2 = activation(np.dot(l1, syn1))
    print('network:')
    A = list(map(int, np.round(l2).T[0]))
    print(A)
    k = np.sum(np.array(A) == np.array(O)) / len(O) * 100
    print('accuracity: {}%'.format(round(k, 2)))


if __name__ == '__main__':
    learn()
