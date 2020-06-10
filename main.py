from brain import *
from generator import X, Y, I, O

X = np.array(X)
Y = np.array([Y]).T
Xs = len(X[0])
Ys = len(Y[0])

N = Network()
N.add(Xs)
N.add(15)
N.add(Ys)
N.train(X, Y, 60000)


A = N.go(I)
A = [int(round(x[0])) for x in A]
print(A)
print('accuracity: {}%'.format(accuracity(A, O)))