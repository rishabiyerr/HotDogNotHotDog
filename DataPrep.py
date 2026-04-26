import pandas as pd
import cv2

def imagePrep(img):
    img = cv2.imread(img, 0)
    df = pd.DataFrame(img.flatten()).T
    return df