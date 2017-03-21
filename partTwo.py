# Neural Networks Demystified
# Part 2: Forward Propagation
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch


## ----------------------- Part 1 ---------------------------- ##
import numpy as np
import matplotlib.pyplot as plt


# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

## ----------------------- Part 2 ---------------------------- ##
verbose=True
doPlot = True

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters == col number
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)

        if verbose:
            print("X")
            print(X)
            print("W1")
            print(self.W1)
            title = "z2 = X . W1"
            print("%s" % title)
            print(self.z2)
            print()
            if doPlot:
                plt.title(title)
                plt.plot(self.z2)
                plt.show()

        self.a2 = self.sigmoid(self.z2)

        if verbose:
            title = "a2 = sigmoid(z2)"
            print(title)
            print(self.a2)
            print()
            if doPlot:
                plt.title(title)
                plt.plot(self.a2)
                plt.show()

        self.z3 = np.dot(self.a2, self.W2)

        if verbose:
            print("W2")
            print(self.W2)
            title = "z3 = a2 . W2"
            print(title)
            print(self.z3)
            print()
            if doPlot:
                plt.title(title)
                plt.plot(self.z3)
                plt.show()

        yHat = self.sigmoid(self.z3)

        if verbose:
            title = "yHat = sigmoid(z3)"
            print(title)
            print(yHat)
            print()
            if doPlot:
                plt.title(title)
                plt.plot(yHat)
                plt.show()

        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
nn = Neural_Network()
nn.forward(X)
