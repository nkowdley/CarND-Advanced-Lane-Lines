## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboard_original]: ./camera_cal/calibration2.jpg "Original Chessboard"
[chessboard_calculations]:./camera_cal/corners_found11.jpg "CV2 Chessboard"
[chessboard_undistorted]:./camera_cal/undistorted_chessboard1.jpg "CV2 Chessboard Undistorted"
[test1]:./test_images/test1.jpg "Original Test Image 1"
[test2]:./output_images/curvature_offset_and_pos3.jpg  "another sample"
[test1_undistorted]:./output_images/undistorted1.jpg "undistorted Image"
[test1_thresholds]:./output_images/thresh_image1.jpg "threshold Image"
[test1_warped]:./output_images/warped1.jpg "warped Image"
[test1_windows]:./output_images/windows1.jpg "window Image"
[test1_lane_lines]:./output_images/lane_lines_drawn1.jpg "lane lines Image"
[test1_curve_and_lines]:./output_images/curvature_offset_and_pos2.jpg "curve Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file camera_calibration.py.

I start by reading in the chessboard image, converting it to grayscale, and then using the cv2 function findChessboardCorners.

For example, this image was provided as a chessboard input:
![alt text][chessboard_original]

And here is the resultant image: 
![alt text][chessboard_calculations]

Here, you can clearly see how we have found the corners on this image.

Each time we successfuly detect all chessboard corners, we add things to 2 arrays:

    1. The result of this function is added to an array where the (x,y) pixel coordinates of each corner that cv2 finds.
    1. Another Array is used to hold the original object points on an x,y,z plane, where we assume a 2d plane.

From these two arrays, we are able to determine the camera calibration and distortion coefficients by mapping these two arrays to each other. cv2 provides us with a function to do this, called calibrate camera.  

Running cv2.undistort on this same input image resulted in:

![alt text][chessboard_undistorted]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][test1]

I used the camera calibration to determine the coefficients needed to convert an image into an undistorted image. For this, I used cv2's undistort function again.  This can be seen in the function undistort in image_gen.py.

The results of that undistortion produces the following image: 
![alt text][test1_undistorted]

A telltale sign that something has changed is the position of the white car in the right hand side of the image.  Notice that part of it seems to disappear when we undistort the image.
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  This can be seen in my code in the thresholds function inside of image_gen.py.  Note that for this function, I actually call into a different file, gradient_and_color_library.py which contains helper functions created during the lessons.

For this step, I used 3 main transforms: 
1. an absolute sobel threshold in the x orientation
1. an absolute sobel threshold in the y orientation
1. a color threshold where I used the hsv and hls color spaces.

Here's an example of my output for this step.  Note that this is the same image from before:
![alt text][test1_thresholds]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I preformed a perspective transform in image_gen.py inside the function warp_image().  This made it seem like we were looking at the road from a bird's eye view. 

To do this, I measured out a trapezoid containing the important information roughly using the image on an image viewer.  From there, I mapped that out onto a rectangle.  My numbers ended up being incredibly wrong, so I ended up using the numbers provided from the Q&A.  Here are the dimensions of my trapezoid:
G_BOT_WIDTH = .76
G_MID_WIDTH = .08
G_HEIGHT_PCT = .62
G_BOTTOM_TRIM = .935 

Here is the same image, but warped using the perspective transform:
![alt text][test1_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For this, I used window sliding and convolutions in order to map out where the lane lines were.  This can be seen in my function line_tracker_windows() inside of image_gen.py.  Note that this function uses the class 'Tracker' which can be found inside of tracker.py

The same image from above, after running the window sliding on it here:  Note that the green is windows where we found lane lines
![alt text][test1_windows]

I then, took the data points from these windows, and used np.polyfit() on them, in order to generate a continuous line of smooth data. I tried using 2nd,3rd, and 4th order polynomials to see which would give me the most accurate line.  This did not seem to change drastically, so I stuck with 4, with the hope that it would be more robust if used in different scenarios. 

Here is what the polynomial fit lane lines look like:
![alt text][test1_lane_lines]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature component can be found in the function get_left_curvature(): inside the image_gen.py file. 

To calculate this, I started with the left lane and used math I found here: https://www.intmath.com/applications-differentiation/8-radius-curvature.php to calculate this. 

The position of the vehicle with respect to center can be found in the functions get_offset_from_center() and get_side_position() in image_gen.py.

For this, I average the left and right lines, and then determine where the car is pixel wise, in relation to the full image.  From there, I use a global variable(pixels per meter or G_XM) to convert the number of pixels from the center of the lane into a usable distance in meters.

Here is what an image looks like overlayed with this information:
![alt text][test1_curve_and_lines]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is what this looks like on a different image:
![alt text][test2]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/output1_tracked.mp4)

---

### Discussion

####Where will your pipeline likely fail?  What could you do to make it more robust?

This project was very difficult to tune, seeing as there are a number of parameters that can be used in thresholding and lane fitting.  My approach is definitely the most basic, and will likely have issues in situations with varying light conditions, as well as very curvy roads as my polynomial function only goes to 4.  It would definitely be more robust if I used some light corrections on my input images.  It would also help my lane fitting lines if I were to use machine learning to determine the appropriate line/polynomial order, instead of always fitting to a 4th degree polynomial
