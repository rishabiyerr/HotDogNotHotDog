import numpy as np

def neuron(x, w, b):
    z = np.dot(x, w)+b
    return 1/(1+np.exp(-z))