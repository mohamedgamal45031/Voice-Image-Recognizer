# Importing OpenCV package
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
import pandas as pd # pandas
import plotly.express as px
import numpy as np

# import cv2_imshow
# Reading the image
# img = cv2.imread('3.jpg', cv2.IMREAD_UNCHANGED)

# # Converting image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# rows,cols,colors = img.shape # gives dimensions for RGB array
# img_size = rows*cols*colors
# img_1D_vector = img.reshape(img_size)
# df = pd.DataFrame(img_1D_vector.reshape(-1, len(img_1D_vector)),columns=[f"feature {i}" for i in range(len(img_1D_vector))])
# pca = decomposition.PCA(n_components=0.85)
# X = pca.fit_transform(df)
# # Loading the required haar-cascade xml classifier file
# face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change

# # Applying the face detection method on the grayscale image
# faces_rect = face_cascade.detectMultiScale(gray_img, 1.1, 9)
# # print(gray_img)
# # Iterating through rectangles of detected faces
# for (x, y, w, h) in faces_rect:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# cv2.imshow('image',img)
# # cv2.imshow('Detected faces', img)

# # img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
# cv2.waitKey(0)

def EuclideanDistance(point1,point2):
    # finding sum of squares
    sum_sq = np.sum(np.square(point1 - point2))
    return np.sqrt(sum_sq)

def image_distance_ratio(df):
    dist1 = EuclideanDistance(df.iloc[0],df.iloc[3])
    dist2 = EuclideanDistance(df.iloc[1],df.iloc[3])
    dist3 = EuclideanDistance(df.iloc[2],df.iloc[3])
    return ((dist1+dist2+dist3)/3)

def imagawy(image1 ,image2,image3,image4):
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img3 = cv2.imread(image3)
    img4 = cv2.imread(image4)
    
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    rows,cols = gray_img1.shape # gives dimensions for RGB array
    img_size1 = rows*cols
    img_1D_vector1 = gray_img1.reshape(img_size1)
    
    rows,cols = gray_img2.shape # gives dimensions for RGB array
    img_size2 = rows*cols
    img_1D_vector2 = gray_img2.reshape(img_size2)

    rows,cols = gray_img3.shape # gives dimensions for RGB array
    img_size3 = rows*cols
    img_1D_vector3 = gray_img3.reshape(img_size3)

    rows,cols = gray_img4.shape # gives dimensions for RGB array
    img_size4 = rows*cols
    img_1D_vector4 = gray_img4.reshape(img_size4)
    
    
    df = pd.DataFrame([img_1D_vector1,img_1D_vector2,img_1D_vector3,img_1D_vector4],columns=[f"feature {i}" for i in range(len(img_1D_vector1))])
    # pca = decomposition.PCA(n_components=59580)
    # X = pca.fit_transform(df)
    # print(df)
    
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces_rect1 = face_cascade.detectMultiScale(gray_img1, 1.1, 9)
    faces_rect2 = face_cascade.detectMultiScale(gray_img2, 1.1, 9)
    faces_rect3 = face_cascade.detectMultiScale(gray_img3, 1.1, 9)
    faces_rect4 = face_cascade.detectMultiScale(gray_img4, 1.1, 9)

    error_ratio = image_distance_ratio(df)

    if(error_ratio>=0.5):
        return 1
    else:
        return 0

# ans =imagawy("1.jpg","2.jpg","3.jpg","4.jpg")
# print(ans)

















# import cv2

# import numpy as np

# import imutils

# # HOGCascade = cv2.HOGDescriptor('data/hogcascades/hogcascade_pedestrians.xml')
# winSize = (64,64)
# blockSize = (16,16)
# blockStride = (8,8)
# cellSize = (8,8)
# nbins = 9
# derivAperture = 1
# winSigma = 4.
# histogramNormType = 0
# L2HysThreshold = 2.0000000000000001e-01
# gammaCorrection = 0
# nlevels = 64
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# image = imutils.resize('1.jpg', width=700)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# clahe = cv2.createCLAHE(clipLimit=15.0,tileGridSize=(8,8))

# gray = clahe.apply(gray)

# winStride = (8,8)

# padding = (16,16)

# scale = 1.05

# meanshift = -1

# (rects, weights) = hog.detectMultiScale(gray, winStride=winStride,

# padding=padding,

# scale=scale,

# useMeanshiftGrouping=meanshift)

# for (x, y, w, h) in rects:

#   cv2.rectangle(image, (x, y), (x+w, y+h), (0,200,255), 2)

# cv2.imshow('Image', image)
