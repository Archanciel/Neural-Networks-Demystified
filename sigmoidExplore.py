import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

tst = sigmoid(np.array([-1,0,1]))
print(tst)
tst = sigmoid(np.random.randn(3,3))
print(tst)
