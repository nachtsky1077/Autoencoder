import numpy as np

class MultiplyGate(object):

    def forward(self, W, X):
        return np.dot(X, W)
    
    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dX, dW