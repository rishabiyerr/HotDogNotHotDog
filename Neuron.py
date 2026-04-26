import numpy as np

def neuron(x, w, b):
    z = np.dot(x/255, w)+b
    return 1/(1+np.exp(-z))