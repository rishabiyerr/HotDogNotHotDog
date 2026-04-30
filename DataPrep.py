import pandas as pd
import cv2
from glob import glob

def imagePrep(img):
    img = cv2.imread(img, 0)
    df = pd.DataFrame(img.flatten()).T
    return df

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