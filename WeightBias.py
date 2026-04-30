import sympy as sp
import numpy as np
import Neuron

def initParams(inputSize):
    np.random.seed(1)
    w = np.random.randn(inputSize,1)*0.1
    b = np.zeros((1,1))
    return w,b

def gradientDescent(x, y, w, b):
    predictions = Neuron.neuron(x, w, b)
    error = predictions - y
    dw = error * np.array(x).reshape(-1, 1)
    db = error
    
    w = w - 0.01 * dw
    b = b - 0.01 * db
    
    return w, b