import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


arr13 = [-1, 0, 1]
tst = sigmoid(np.array(arr13))
print("arr13")
print(arr13)
print("after sigmoid")
print(tst)
matrix33 = np.random.randn(3, 3)
print("matrix33")
print(matrix33)
print("after sigmoid")
tst = sigmoid(matrix33)
print(tst)
