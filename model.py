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

ft = np.load('trainData.npy')
lbl = np.load('trainLabel.npy')

# print ft.shape[1]

k = 5

nbr = kNN(n_neighbors=k, weights='distance').fit(ft,lbl)

np.save('model.npy',nbr)