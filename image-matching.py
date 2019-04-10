import numpy as np 
import cv2
import os 
from matplotlib import pyplot as plt 

# Initialize directory
directoryObj = "object"
directoryScn = "scene"

# Initialize array
obj = []
scene = []
title = []
result = []

# Read images in folder and Insert them to Array
for filename in os.listdir(directoryObj):
    obj.append(cv2.imread(directoryObj+"/"+filename))

    title.append(filename)

for filename in os.listdir(directoryScn):
    scene.append(cv2.imread(directoryScn+"/"+filename, 0))

for i in range(len(obj)):
    # Gaussian Blur Filter
    obj[i] = cv2.GaussianBlur(obj[i], (11,11), 0)

    # Convert each images in obj to GRAY
    # because histogram equalize can be applied on GRAYSCALE image
    obj[i] = cv2.cvtColor(obj[i], cv2.COLOR_BGR2GRAY)

    # Histogram Equalize
    obj[i] = cv2.equalizeHist(obj[i])

    # SURF
    surf = cv2.xfeatures2d.SURF_create()

    kp_obj, des_obj = surf.detectAndCompute(obj[i], None)
    kp_scene, des_scene = surf.detectAndCompute(scene[i], None)

    # Initialize parameters for FLANN
    index_params = dict(algorithm=0)
    search_params = dict(checks=50)

    # FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_obj, des_scene, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for idx, (m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[idx] = [1,0]

    result.append(cv2.drawMatchesKnn(obj[i], kp_obj, scene[i], kp_scene, matches, None, matchColor = [0,255,0], singlePointColor = [255,0,0], matchesMask = matchesMask))

# Show Result images
for i in range(len(result)):
    plt.subplot(1,len(result),i+1), plt.imshow(result[i])

plt.show()

cv2.waitKey(0)