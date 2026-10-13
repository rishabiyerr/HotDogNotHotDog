import numpy as np
import DataPrep
import WeightBias
import cv2
import random

xTrain, yTrain = DataPrep.loadDataSet('DataSet/train')
xTest,  yTest  = DataPrep.loadDataSet('DataSet/test')

np.random.seed(1)
n_features = DataPrep.imagePrep(xTrain[0]).shape[1]

params = {}
params['W1'], params['b1'] = WeightBias.initParams(n_features, 16)
params['W2'], params['b2'] = WeightBias.initParams(16, 1)

indices = np.arange(len(xTrain))

for epoch in range(25):
    np.random.shuffle(indices)
    rotation_options = [
        None, 
        cv2.ROTATE_90_CLOCKWISE, 
        cv2.ROTATE_180, 
        cv2.ROTATE_90_COUNTERCLOCKWISE
    ]
    for step, i in enumerate(indices):
        angle = random.choice(rotation_options)
        x = DataPrep.imagePrep(xTrain[i], angle=angle)
        y = np.array(yTrain[i]).reshape(1, 1)
        a2, cache = WeightBias.forward(x, params)
        params = WeightBias.gradientDescent(y, params, cache)
        loss = -(y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))
        if step % 100 == 0:
            print(f"epoch {epoch}, step {step}, loss {loss.item():.4f}")

correct = 0
for i in range(len(xTest)):
    x = DataPrep.imagePrep(xTest[i])
    y = yTest[i]
    a4, _ = WeightBias.forward(x, params)
    pred = 1 if a4.item() > 0.5 else 0
    if pred == y:
        correct += 1
print(f"Test accuracy: {correct / len(xTest):.3f}")

train_correct = 0
for i in range(len(xTrain)):
    x = DataPrep.imagePrep(xTrain[i])
    y = yTrain[i]
    a2, _ = WeightBias.forward(x, params)
    pred = 1 if a2.item() > 0.5 else 0
    if pred == y:
        train_correct += 1
print(f"Train accuracy: {train_correct / len(xTrain):.3f}")