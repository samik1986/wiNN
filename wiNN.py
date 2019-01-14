import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.neighbors import KNeighborsRegressor as kNNr
from PIL import Image
import cv2
import os, sys
import glob
import pandas as pd
from scipy.spatial import distance

imgPath = '/Users/banerjee/mnt/bnbdev1/PMD2057/'
outPath = '/Users/banerjee/Documents/ICML/GfpANNO/OutANNO/'
nbr = np.load('model.npy')
k = 5

def wt(x, x_i, x_k, delta):
    w = np.power(distance.euclidean(x, x_k) / distance.euclidean(x, x_i), delta)
    return w

def y_hat(x, nbr, k):
    num = 0
    den = 0
    nnx = nbr.kneighbors(x, return_distance=False)
    for i in range(1, k):
        y_i = nbr.predict(nnx[i])
        num = num + (wt(x, nnx[i,], nnx[k,], 2) * y_i)
        den = den + wt(x, nnx[i,], nnx[k,], 2)
    return np.sign(num/den)

files = [os.path.join(imgPath, x) for x in os.listdir(imgPath)]
for testFile in files:
    testImg = cv2.imread(testFile)
    outImg = np.zeros([np.size(testImg, 0), np.size(testImg, 0)])
    for row in range(1, np.size(testImg, 0) - 2, 3):
        for col in range(1, np.size(testImg, 1) - 2, 3):
            subImg = testImg[row: row + 5, col: col + 5]
            linImg = subImg.flatten()
            val = y_hat(linImg, nbr, k)
            if val < 0 :
                classLbl = 0
            else:
                classLbl = 1
            outImg[row + 2, col + 2] = classLbl
    cv2.imwrite(os.path.join(outPath,testImg[1:59]))