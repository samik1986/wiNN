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

annoPath = '/Users/banerjee/Documents/ICML/GfpANNO/TIFANNO/'
imgPath = '/Users/banerjee/mnt/bnbdev1/PMD2057/'
# filename, file_extension = os.path.splitext(annoPath)
lbl = []
ft = []
prevImgFile = 'hello'
files = [os.path.join(annoPath, x) for x in sorted(os.listdir(annoPath))]
for annoFile in files:
    print annoFile
    # df = pd.read_csv(annoFile, header=None)
    annoImg = cv2.imread(annoFile, cv2.IMREAD_UNCHANGED)
    # print np.size(annoImg)
    # print annoCrop
    filename, file_extension = os.path.splitext(annoFile)
    subName = filename[47:61]
    # print subName, filename[68:72], filename[73:]
    xC = int(filename[68:72])
    yC = int(filename[73:])
    # print glob(glob(imgPath + '*'+ subName + '*.jp2'))

    for imgFile in glob.glob(imgPath + '*'+ subName + '*.jp2'):
        # print imgFile
        # print prevImgFile
        if imgFile != prevImgFile:
            imgOrig = cv2.imread(imgFile)
            img = imgOrig[:,:,1]
            img = np.squeeze(img)
            # print 'In check'
        imgCrop = img[xC : xC + 500, yC : yC + 500]
        # print np.size(imgCrop,0)
        for row in range(1, np.size(imgCrop,0) - 5, 3):
            for col in range(1, np.size(imgCrop, 1) - 5, 3):
                classId = annoImg[row + 2, col + 2]
                # print classId
                if classId == 0:
                    classId = -1
                else :
                    classId = 1
                lbl.append(classId)
                subImg = imgCrop[row : row + 5, col : col + 5]
                linImg = subImg.flatten()
                # print np.size(linImg)
                ft.append(linImg)
                # print np.size(ft,axis=1)
        prevImgFile = imgFile
np.save('trainData.npy',ft)
np.save('trainLabel.npy', lbl)

