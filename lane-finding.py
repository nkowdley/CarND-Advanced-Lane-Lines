#!/usr/bin/env python 
"""
A python script for the Udacity Project Advanced Lane Lines
"""
import numpy as np
import cv2
import glob
import pickle
from pprint import pprint
from tracker import Tracker

# The following globals are designed to make development/debug easier.  In a more real world enviornment,
# Both of these would be turned off.
DEBUG = 1 # A switch for print statements.  Turn off to make the script not print out anything
OUTPUT_STEPS = 1 # A switch for writing out files.  Turn on to output images from each individual step.

# The following globals are hyper parameters, designed for tuning the pipeline:
G_WINDOW_WIDTH = 25
G_WINDOW_HEIGHT = 80
G_MARGIN = 25
G_YM = 10/720 #( 10 meters = 720 pixels)
G_XM = 4/384 #( 4 meters = 384 pixels)

##########################################################################
# Utility Functions
##########################################################################
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
                write_name = "./camera_cal/corners_found" + str(idx) + ".jpg"
                cv2.imwrite(write_name,img)
                if DEBUG == 1:
                    print("Writing File: " + write_name)
        else:   
            if DEBUG == 1:
                print("file: " + fname + " with idx: " + str(idx) + "could not be processed")
    
    # Load a reference image to get the image size
    img = cv2.imread("./camera_cal/calibration1.jpg")
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, img_size, None, None)
    return mtx, dist

# From the Udactiy Lessons, we have some helpful functions.
# I have left the comments in to help explain what is happening here
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return s_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    sobelxy = np.absolute(np.sqrt(sobelx**2 + sobely**2))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return s_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    s_binary = np.zeros_like(direction)
    s_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return s_binary

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
    
def hsv_select(img, thresh=(0,255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel >= thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

def color_threshold(img, sthresh=(0,255), vthresh=(0,255)):
    # We do this first part to get the output array to be the correct size
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(hls_select(img, sthresh) == 1) & (hsv_select(img, vthresh)==1)] = 1
    return binary_output

#TODO: Adjust this function
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

def undistort(mtx, dist, image):
    return cv2.undistort(image, mtx, dist, None, mtx)

##########################################################################
# The following is a pipeline, designed for use with the test files
##########################################################################
def test_pipeline(mtx, dist, glob_image_path):
    images = glob.glob(glob_image_path)
    # sort the array so that it is easier to determine which images map to which
    images.sort()
    for idx, fname in enumerate(images):
        if DEBUG == 1:
            print("Looking at file: " + fname + " with idx of: " + str(idx+1))
        img = cv2.imread(fname)
        # undistort
        img = undistort(mtx, dist, img)
        # color/gradient thresholds
        thresh_image = np.zeros_like(img[:,:,0])
        gradx = abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255) # Orginal = 12
        grady = abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255) # Original = 25
        c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
        thresh_image[(gradx == 1) & (grady == 1) | (c_binary == 1 )] = 255
        if OUTPUT_STEPS == 1:
            write_name = './test_images/preprocessed_image' + str(idx+1) + '.jpg'
            print("Writing file: " + write_name)
            cv2.imwrite(write_name, thresh_image)
        img_size = (img.shape[1], img.shape[0])
        # TODO: GLobalize these
        bot_width = .76 #% for bottom trapizoid height Original:.76
        mid_width = .08 # % for middle trap height Original .08
        height_pct = .62 # % for trapizoid height Original .62
        bottom_trim = .935 # % from top to bottom to ignore car Original .935
        src = np.float32([[img.shape[1] * (.5 - mid_width/2), img.shape[0] * height_pct], [img.shape[1] * (.5 + mid_width/2), img.shape[0] * height_pct], [img.shape[1] * (.5 + bot_width/2), img.shape[0] * bottom_trim],[img.shape[1] * (.5 - bot_width/2), img.shape[0] * bottom_trim]])
        # TODO: Globalize this
        offset = img_size[0] * .25
        dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
        M = cv2.getPerspectiveTransform(src,dst)
        M_inverse = cv2.getPerspectiveTransform(dst,src) # used to do the opposite transform
        warped = cv2.warpPerspective(thresh_image, M, img_size, flags=cv2.INTER_LINEAR)
        if OUTPUT_STEPS == 1:
            write_name = './test_images/warped' + str(idx+1) + '.jpg'
            print("Writing file: " + write_name)
            cv2.imwrite(write_name, warped)
        # Set up our parameters for the line tracker
        window_width = G_WINDOW_WIDTH
        window_height = G_WINDOW_HEIGHT
        margin = G_MARGIN
        ym = G_YM
        xm = G_XM
        line_tracker = Tracker(window_width = window_width, window_height = window_height, margin = margin, y_m = ym, x_m = xm)
        window_centeroids = line_tracker.find_window_centeroids(warped)
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        leftx = []
        rightx = []
        # Go Through Each Level to draw the windows
        for level in range(0, len(window_centeroids)):
            leftx.append(window_centeroids[level][0])
            rightx.append(window_centeroids[level[1]])
            l_mask = window_mask(window_width, window_height, warped, window_centeroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centeroids[level][1], level)
            l_points[ (l_points == 255) | ((l_mask == 1)) ] = 255
            r_points[ (r_points == 255) | ((r_mask == 1)) ] = 255
        if OUTPUT_STEPS == 1:
            # Add left and right window pixels together
            template = np.array(r_points+l_points, np.uint8)
            # Create Zero color channels
            zero_channel = np.zeros_like(template)
            # make window pixels green by only using the G Channel
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
            # convert the original channels to rgb
            warpage = np.array(cv2.merge((warped,warped,warped)), np.uint8)
            # overlay the results
            result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)
            write_name = './test_images/windows' + str(idx+1) + '.jpg'
            print("Writing file: " + write_name)
            cv2.imwrite(write_name, result)
        yvals = range(0,warped.shape[0])
        res_yvals = np.arrange(warped.shape[0] - (window_height/2), 0, -window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        # Polyfit finds the coefficients, using a degree of 2
        # ax^2 + bx + c
        # First polyfit the left
        left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)
        # Next Polyfit the right
        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0]*yvals*yvals + right_fit[1]* yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        left_lane = 
        right_lane = 
        middle_marker = 

if __name__ == "__main__":
    # This project requires us to do the following steps:
    # 1) Camera Calibration
    # 2) Undistortion
    # 3) Color and Gradient Threshold
    # 4) Warp using Perspective Transform

    # Step 1: Calibrate Camera.  Note that our chess board is 6x9
    mtx, dist = camera_calibration(6, 9 ,'./camera_cal/calibration*.jpg')
    # Step 2: Undistort the images and apply thresholds
    test_pipeline(mtx, dist, './test_images/test*.jpg')


