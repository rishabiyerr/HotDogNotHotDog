import numpy as np
import DataPrep
import Neuron

def initParams(inputSize):
    np.random.seed(1)
    w = np.random.randn(inputSize,1)*0.1
    b = np.zeros((1,1))
    return w,b

x = DataPrep.imagePrep('TestImage(1).jpg')
w, b = initParams(x.shape[1])
print(Neuron.neuron(x, w, b))