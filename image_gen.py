#!/usr/bin/env python 
"""
A python script for the Udacity Project Advanced Lane Lines

This file is meant to be used with images, like the test images provided
"""
import numpy as np
import cv2
import glob
import pickle
from pprint import pprint
# import my modules
from tracker import Tracker
from camera_calibration import *
from gradient_and_color_libary import *

# The following globals are designed to make development/debug easier.  In a more real world enviornment,
# Both of these would be turned off.
DEBUG = 1 # A switch for print statements.  Turn off to make the script not print out anything
OUTPUT_STEPS = 1 # A switch for writing out files.  Turn on to output images from each individual step.

# Program Globals:
G_TEST_IMAGE_FOLDER = './test_images/test*.jpg'
# Offset for warping
G_OFFSET = .25
# Trapezoid Globals
G_BOT_WIDTH = .76 #% for bottom trapizoid height Original:.76
G_MID_WIDTH = .08 # % for middle trap height Original .08
G_HEIGHT_PCT = .62 # % for trapizoid height Original .62
G_BOTTOM_TRIM = .935 # % from top to bottom to ignore car Original .935

# The following are used for Tracker() and associated functions with tracker:
G_WINDOW_WIDTH = 25
G_WINDOW_HEIGHT = 80
G_MARGIN = 25
G_YM = 10/720 #( 10 meters = 720 pixels)
G_XM = 4/384 #( 4 meters = 384 pixels)

def undistort(mtx, dist, image, idx):
    """
    A Simple Wrapper Around the openCV call to undistort
    """
    undistort = cv2.undistort(image, mtx, dist, None, mtx)
    if OUTPUT_STEPS == 1:
        write_name = "./output_images/undistorted" + str(idx) + ".jpg" # This path is hardcoded here
        cv2.imwrite(write_name,undistort)
        if DEBUG == 1:
            print("Writing File: " + write_name)

    return undistort


def thresholds(img, idx):
    """
    Applies our gradient and color thresholds
    The functions used in here are found in gradient_and_color_library

    Note that these thresholds are hard-coded in.
    """
    # color/gradient thresholds
    thresh_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255) # Orginal = 12
    grady = abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255) # Original = 25
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    thresh_image[(gradx == 1) & (grady == 1) | (c_binary == 1 )] = 255
    if OUTPUT_STEPS == 1:
        write_name = './output_images/preprocessed_image' + str(idx+1) + '.jpg'
        print("Writing file: " + write_name)
        cv2.imwrite(write_name, thresh_image)
    return thresh_image

def warp_image(img, img_size, idx, offset_pct = G_OFFSET):
    """
    This function warps our image so that we can see a bird's eye view of the road
    """
    src = np.float32([[img.shape[1] * (.5 - G_MID_WIDTH/2), img.shape[0] * G_HEIGHT_PCT], [img.shape[1] * (.5 + G_MID_WIDTH/2), img.shape[0] * G_HEIGHT_PCT], [img.shape[1] * (.5 + G_BOT_WIDTH/2), img.shape[0] * G_BOTTOM_TRIM],[img.shape[1] * (.5 - G_BOT_WIDTH/2), img.shape[0] * G_BOTTOM_TRIM]])
    # TODO: Globalize this
    offset = img_size[0] * offset_pct
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src,dst)
    M_inverse = cv2.getPerspectiveTransform(dst,src) # used to do the opposite transform
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    if OUTPUT_STEPS == 1:
        write_name = './output_images/warped' + str(idx+1) + '.jpg'
        print("Writing file: " + write_name)
        cv2.imwrite(write_name, warped)
    return M, M_inverse, warped

def line_tracker_windows(warped, idx, window_width = G_WINDOW_WIDTH, window_height = G_WINDOW_HEIGHT, margin = G_MARGIN, y_m = G_YM, x_m = G_XM):
    """
    This function finds the lines using window sliding
    """
    # Set up LineTracker
    line_tracker = Tracker(window_width = window_width, window_height = window_height, margin = margin, y_m = y_m, x_m = x_m)
    window_centroids = line_tracker.find_window_centroids(warped)
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    leftx = []
    rightx = []
    # Go Through Each Level to draw the windows
    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
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
        write_name = './output_images/windows' + str(idx+1) + '.jpg'
        print("Writing file: " + write_name)
        cv2.imwrite(write_name, result)

    yvals = range(0,warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (G_WINDOW_HEIGHT/2), 0, -G_WINDOW_HEIGHT)

    return leftx, rightx, yvals, res_yvals

def get_lane_lines(res_yvals, yvals, leftx, rightx, window_width = G_WINDOW_WIDTH):
    """
    This function finds the lane line using polyfit
    Polyfit finds the coefficients, for example:
    using a degree of 2
    ax^2 + bx + c
    left_fit is now an array with [a,b,c]
    This function uses a 4th degree polynomial
    """
    left_fit = np.polyfit(res_yvals, leftx, 4)
    left_fitx = left_fit[0] * yvals * yvals * yvals * yvals + left_fit[1] * yvals * yvals * yvals + left_fit[2]* yvals * yvals + left_fit[3] * yvals + left_fit[4]
    left_fitx = np.array(left_fitx, np.int32)
    # Do the same polyfit on the right
    # Next Polyfit the right
    right_fit = np.polyfit(res_yvals, rightx, 4)
    right_fitx = right_fit[0] * yvals * yvals * yvals * yvals + right_fit[1] * yvals * yvals * yvals + right_fit[2]* yvals * yvals + right_fit[3] * yvals + right_fit[4]
    right_fitx = np.array(right_fitx, np.int32)
    # package up left lane and right lane
    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0), np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),  np.concatenate((yvals,yvals[::-1]),axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),  np.concatenate((yvals,yvals[::-1]),axis=0))), np.int32)

    return left_lane, right_lane, inner_lane, left_fitx, right_fitx

def get_offset_from_center(left_fitx, right_fitx, warped, xm = G_XM):
    camera_center = (left_fitx[-1] + right_fitx[-1])/2 
    center_diff = (camera_center-warped.shape[1]/2) * G_XM
    return center_diff

def get_side_position(center_diff):
    if center_diff <= 0:
        side_pos = 'right'
    else:
        side_pos = 'left'
    return side_pos

def get_left_curvature(res_yvals, yvals, leftx, ym = G_YM, xm = G_XM):
    """
    This function finds the curvature of the left line.
    Curvature Math taken from:
    https://www.intmath.com/applications-differentiation/8-radius-curvature.php
    """
    #calculate the left curvature
    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym, np.array(leftx,np.float32) * xm, 2)
    curve_rad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym + curve_fit_cr[1])**2)**1.5) / np.absolute(2 * curve_fit_cr[0])
    return curve_rad


#TODO: Adjust this function
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

##########################################################################
# The following is a pipeline, designed for use with images
# This will get called when attempting to run this script as a script
# ie ./image_gen.py
##########################################################################
def image_pipeline(mtx, dist, img, idx = 0):
    # undistort the image
    img = undistort(mtx, dist, img, idx)

    # Apply our color and graident thresholds
    thresh_image = thresholds(img, idx)
    img_size = (img.shape[1], img.shape[0])

    # Warp the image
    M, M_inverse, warped = warp_image(thresh_image, img_size, idx)

    # Use window sliding to find the lines
    leftx, rightx, yvals, res_yvals = line_tracker_windows(warped, idx)

    left_lane, right_lane, inner_lane, left_fitx, right_fitx = get_lane_lines(res_yvals, yvals, leftx, rightx)

    # The following Code Draws the Lane Lines found above onto an image.
    # Draw the lane lines
    road = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color = [255, 0, 0])
    cv2.fillPoly(road, [inner_lane], color = [0, 255, 0])
    cv2.fillPoly(road, [right_lane], color = [0, 0, 255])
    road_warped = cv2.warpPerspective(road, M_inverse, img_size, flags=cv2.INTER_LINEAR)
    # isolate the background so that we can more clearly see the lines
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road_bkg,[left_lane], color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane], color=[255,255,255])
    road_warped_bkg = cv2.warpPerspective(road_bkg, M_inverse, img_size, flags=cv2.INTER_LINEAR)
    # Create the image to output
    lane_lines_drawn_base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    lane_lines_drawn = cv2.addWeighted(lane_lines_drawn_base, 1.0, road_warped, 1.0, 0.0)
    if OUTPUT_STEPS == 1:
        write_name = './output_images/lane_lines_drawn' + str(idx+1) + '.jpg'
        print("Writing file: " + write_name)
        cv2.imwrite(write_name, lane_lines_drawn)

    # calculate the offset of the car
    center_diff = get_offset_from_center(left_fitx, right_fitx, warped)
    # get the position of the offset(left or right of center)
    side_pos = get_side_position(center_diff)
    # Get the curvature of the left lane line
    curve_rad = get_left_curvature(res_yvals, yvals, leftx)
    if DEBUG == 1:
        print("Curve Radius is " + str(curve_rad))
        print("Side Position is " + side_pos)
        print("Center offset is " + str(center_diff))
    # Add text to the image we are returning
    cv2.putText(lane_lines_drawn, 'Radius of Curvature is ' + str(round(abs(curve_rad),3)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2 )
    cv2.putText(lane_lines_drawn, 'Vehicle is ' + str(round(abs(center_diff),3)) + ' meters ' + side_pos + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2 )
    if OUTPUT_STEPS == 1:
        write_name = './output_images/curvature_offset_and_pos' + str(idx+1) + '.jpg'
        print("Writing file: " + write_name)
        cv2.imwrite(write_name, lane_lines_drawn)
    return lane_lines_drawn

if __name__ == "__main__":
    # Calibrate Camera.  Note that our chess board is 6x9
    mtx, dist = camera_calibration(6, 9 ,'./camera_cal/calibration*.jpg')
    #  Run Our Pipeline on each image
    images = glob.glob(G_TEST_IMAGE_FOLDER)
    # sort the array so that the images are in order of their number
    images.sort()
    for idx, fname in enumerate(images):
        if DEBUG == 1:
            print("Looking at file: " + fname + " with idx of: " + str(idx+1))
        # Read in the file
        img = cv2.imread(fname)
        # Pass it to the pipeline
        image_pipeline(mtx, dist, img, idx+1)