#!/usr/bin/env python 
"""
A python script for the Udacity Project Advanced Lane Lines
This library contains the tracker class, which tracks the lines on the road
"""
import numpy as np
import cv2
from pprint import pprint

class Tracker():
    def __init__(self, window_width, window_height, margin, y_m = 1, x_m = 1 , smooth_factor = 15):
        # list that stores all the past center set values which will be used for smoothing
        self.recent_centers = [] 
        # The following 3 parameters are used for our window searching algorithm
        # window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = window_width 
        # window pixel height of the center values, used to count pixels inside center windows to determine curve values
        self.window_height = window_height
        # pixel distance in both directions to slide (left_window + right_window) template for searching
        # AKA Padding
        self.margin = margin

        self.vertical_meters_per_pixel = y_m
        self.horizontal_meters_per_pixel = x_m
        self.smooth_factor = 15

    #the main tracking function for finding and storing lane segment positions
    def find_window_centeroids(self, warped):
        window_centeroids = []
        window = np.ones(self.window_width) # create window template that we will use for convolutions
        # Start with the bottom quarter of the image
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis = 0)
        l_center = np.argmax(np.convolve(window,l_sum)) - self.window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis = 0)
        r_center = np.argmax(np.convolve(window,r_sum)) - self.window_width/2 + int(warped.shape[1]/2)
        # add the first layer
        window_centeroids.append((l_center, r_center))
        # Go through each layer looking for max pixel locations
        # Note that warped.shape[0]/self.window_height is how many vertical slices we have
        for level in range(1, (int)(warped.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0] - (level+1) * self.window_height):int(warped.shape[0] - level * self.window_height),:], axis = 0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centeroid by using previous left center as a reference
            # Use self.window_width/2 as offset because conv signal reference is at right side of window, not left
            offset = self.window_width/2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the right centeroid
            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # add this layer
            window_centeroids.append((l_center, r_center))
        self.recent_centers.append(window_centeroids)
        return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)

