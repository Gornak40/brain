import numpy as np


def sigm(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def relu(x, deriv=False):
    if deriv:
        return x >= 0
    return np.maximum(x, 0)


def accuracity(A, B):
    return round(np.sum(np.array(A) == np.array(B)) / len(A) * 100, 2)


class Network:
    def __init__(self):
        self.lay = list()
        self.syn = list()
        self.num = list()
        self.n = 0
    
    def add(self, _num):
        self.n += 1
        self.num.append(_num)
    
    def init(self):
        self.lay = [None] * self.n
        self.syn = [None] * (self.n - 1)
        for ind in range(self.n - 1):
            self.syn[ind] = 2 * np.random.rand(self.num[ind], self.num[ind + 1]) - 1
    
    def go(self, X):
        self.lay[0] = X
        for ind in range(1, self.n):
            self.lay[ind] = sigm(np.dot(self.lay[ind - 1], self.syn[ind - 1]))
        return self.lay[self.n - 1]
    
    def train(self, inp, out, eras):
        self.init()
        for era in range(eras):
            self.go(inp)
            err = [None] * self.n
            dlt = [None] * self.n
            err[self.n - 1] = out - self.lay[self.n - 1]
            dlt[self.n - 1] = err[self.n - 1] * sigm(self.lay[self.n - 1], deriv=True)
            for ind in reversed(range(1, self.n - 1)):
                err[ind] = np.dot(dlt[ind + 1], self.syn[ind].T)
                dlt[ind] = err[ind] * sigm(self.lay[ind], deriv=True)
            if era % 10000 == 0:
                print('error:', np.mean(np.abs(err[self.n - 1])))
            for ind in reversed(range(self.n - 1)):
                self.syn[ind] += np.dot(self.lay[ind].T, dlt[ind + 1])
                
