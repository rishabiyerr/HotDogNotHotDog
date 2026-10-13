import numpy as np
import Neuron

def initParams(inputSize, outputSize):
    w = np.random.randn(inputSize, outputSize) * np.sqrt(1 / inputSize)
    b = np.zeros((1, outputSize))
    return w, b

def forward(x, params):
    a1 = Neuron.neuron(x,  params['W1'], params['b1'])
    a2 = Neuron.neuron(a1, params['W2'], params['b2'])
    cache = {'x': x, 'a1': a1, 'a2': a2}
    return a2, cache

def gradientDescent(y, params, cache, lr=0.001, reg=0.001):
    x, a1, a2 = cache['x'], cache['a1'], cache['a2']

    dz2 = a2 - y
    dW2 = a1.T @ dz2 + reg * params['W2']
    db2 = dz2

    dz1 = (dz2 @ params['W2'].T) * a1 * (1 - a1)
    dW1 = x.T @ dz1 + reg * params['W1']
    db1 = dz1

    params['W1'] -= lr * dW1; params['b1'] -= lr * db1
    params['W2'] -= lr * dW2; params['b2'] -= lr * db2

    return params