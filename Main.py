import numpy as np
import DataPrep
import Neuron
import WeightBias

xTrain, yTrain = DataPrep.loadDataSet('DataSet/train')
xTest,  yTest  = DataPrep.loadDataSet('DataSet/test')

x = DataPrep.imagePrep(xTrain[1])
w, b = WeightBias.initParams(x.shape[1])
w1, b1 = WeightBias.initParams(64)
w2, b2 = WeightBias.initParams(16)
w3, b3 = WeightBias.initParams(4)
epoch = 50
for f in range (epoch):
    for i in range(len(xTrain)):
        x = DataPrep.imagePrep(xTrain[i])

        inputLayer = []
        for y in range(64):
            inputLayer.append(Neuron.neuron(x, w, b))


        layerTwo = []
        for y in range (0, 64, 4):
            h = max(inputLayer[y], inputLayer[y+1], inputLayer[y+2], inputLayer[y+3])
            layerTwo.append(Neuron.neuron(h, w1, b1))

        layerThree = []
        for y in range (0, 16, 4):
            h = max(layerTwo[y], layerTwo[y+1], layerTwo[y+2], layerTwo[y+3])
            layerThree.append(Neuron.neuron(h, w2, b2))

        output = []
        for y in range (0, 4, 4):
            h = max(layerThree[y], layerThree[y+1], layerThree[y+2], layerThree[y+3])
            output.append(Neuron.neuron(h, w3, b3))
        print(output)
        w, b = WeightBias.gradientDescent(x, yTrain[i], w, b)
        w1, b1 = WeightBias.gradientDescent(h, yTrain[i], w1, b1)
        w2, b2 = WeightBias.gradientDescent(h, yTrain[i], w2, b2)
        w3, b3 = WeightBias.gradientDescent(h, yTrain[i], w3, b3)