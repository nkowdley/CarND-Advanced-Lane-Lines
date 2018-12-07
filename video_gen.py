#!/usr/bin/env python 
"""
A python script for the Udacity Project Advanced Lane Lines

This file is meant to be used with videos
"""
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import numpy as np
import cv2
import glob
import pickle
from pprint import pprint
# import my modules
from tracker import Tracker
from camera_calibration import *
from gradient_and_color_libary import *
from image_gen import * #import all the functions we wrote for finding lane lines on an image

# The following globals are designed to make development/debug easier.  In a more real world enviornment,
# Both of these would be turned off.
DEBUG = 1 # A switch for print statements.  Turn off to make the script not print out anything
OUTPUT_STEPS = 1 # A switch for writing out files.  Turn on to output images from each individual step.

class ImagePipeline:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
    def __call__(self, image):
        return image_pipeline(mtx, dist, image)


def video_pipeline(mtx, dist, input_video_filename, output_video_filename):
    # write the video out to a file
    clip1 = VideoFileClip(input_video_filename)
    video_clip = clip1.fl_image(ImagePipeline(mtx,dist))
    video_clip.write_videofile(output_video_filename, audio = False)

if __name__ == "__main__":
    # Calibrate Camera.  Note that our chess board is 6x9
    mtx, dist = camera_calibration(6, 9 ,'./camera_cal/calibration*.jpg')
    # Run Our Pipeline
    video_pipeline(mtx, dist, 'project_video.mp4', 'output1_tracked.mp4')
