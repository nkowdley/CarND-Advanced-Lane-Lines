#!/usr/bin/env python 
"""
A python library for the Udacity Project Advanced Lane Lines

This library is designed to abstract away the camera calibration component of
this project.
"""
import numpy as np
import cv2
import glob
import pickle
from pprint import pprint

# The following globals are designed to make development/debug easier.  In a more real world enviornment,
# Both of these would be turned off.
DEBUG = 1 # A switch for print statements.  Turn off to make the script not print out anything
OUTPUT_STEPS = 1 # A switch for writing out files.  Turn on to output images from each individual step.

def camera_calibration(rows, cols, glob_image_path):
    if DEBUG == 1:
        print("##### Calibrating Camera #####")
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    # TODO: Parmaeterize this call.
    images = glob.glob(glob_image_path)
    images.sort()
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
            if OUTPUT_STEPS == 1:
                cv2.drawChessboardCorners(img, (rows,cols), corners, ret)
                write_name = "./camera_cal/corners_found" + str(idx) + ".jpg" # This path is hardcoded here
                cv2.imwrite(write_name,img)
                if DEBUG == 1:
                    print("Writing File: " + write_name)
        else:   
            if DEBUG == 1:
                print("file: " + fname + " with idx: " + str(idx) + " could not be processed")
    
    # Load a reference image to get the image size
    img = cv2.imread("./camera_cal/calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, img_size, None, None)
    return mtx, dist

def chess_undistort(mtx, dist, image, idx):
    """
    A Simple Wrapper Around the openCV call to undistort
    """
    undistort = cv2.undistort(image, mtx, dist, None, mtx)
    if OUTPUT_STEPS == 1:
        write_name = "./camera_cal/undistorted_chessboard" + str(idx) + ".jpg" # This path is hardcoded here
        cv2.imwrite(write_name,undistort)
        if DEBUG == 1:
            print("Writing File: " + write_name)


if __name__ == "__main__":
    # Calibrate Camera.  Note that our chess board is 6x9
    mtx, dist = camera_calibration(6, 9 ,'./camera_cal/calibration*.jpg')
    #  Undistort the calibration image and write it out to a file
    image = cv2.imread('./camera_cal/calibration2.jpg')
    chess_undistort(mtx, dist, image, 1)