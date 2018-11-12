#!/usr/bin/env python 
"""
A python script for the Udacity Project Advanced Lane Lines
"""
import numpy as np
import cv2
import glob
import pickle

DEBUG = 1 
# Note this is a 6x9 checkerboard
def camera_calibration(rows=6, cols=9):
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    # TODO: Parmaeterize this call.
    images = glob.glob('./camera_cal/calibration*.jpg')

    for idx,fname in enumerate(images):
        #read the file
        img = cv2.imread(fname)
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #find the corners using cv2
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret == True:
            if DEBUG == 1:
                print("Processing Checkboard image: " + fname + " with idx: " + str(idx))
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (rows,cols), corners, ret)
            write_name = "./camera_cal/corners_found" + str(idx) + ".jpg"
            cv2.imwrite(write_name,img)
            if DEBUG == 1:
                print("Writing File: " + write_name)
    # Load a reference image to get the image size
    # TODO: move this call into the for loop
    img = cv2.imread("./camera_cal/calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])
    #TODO: do not pickle this
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, img_size, None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"]= dist
    pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))

camera_calibration()