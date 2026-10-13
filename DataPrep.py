import numpy as np
import cv2
from glob import glob

IMG_SIZE = 64

def imagePrep(img_path, angle=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if angle is not None:
        img = cv2.rotate(img, angle)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = img.flatten().astype(np.float32) / 255.0
    return x.reshape(1, -1)

def loadDataSet(folder):
    images = []
    labels = []

    for path in glob(f'{folder}/hot_dog/*.jpg'):
        images.append(path)
        labels.append(1)

    for path in glob(f'{folder}/not_hot_dog/*.jpg'):
        images.append(path)
        labels.append(0)

    return images, labels