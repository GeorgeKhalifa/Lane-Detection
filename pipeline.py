import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from helper_functions import *
from line import *

l_line = Line()
r_line = Line()

w=1280
h=720
src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

# Define a function that thresholds the B-channel of LAB
# Use exclusive lower bound (>) and inclusive upper (<=), OR the results of the thresholds (B channel should capture
# yellows)
def lab_bthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048 / 100  # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7 / 378  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1] / 2
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist

# Define the complete image processing pipeline, reads raw image and returns binary image with lane lines identified
# (hopefully)
def pipeline(img):
    # img_unwarp=canny_edge_detector(img)
    # img_unwarp_crop=ROI_mask(img_unwarp)
    # Perspective Transform
    img_unwarp_crop, M, Minv = unwarp(img, src, dst)
    # Sobel Absolute (using default parameters)
    # img_sobelAbs = abs_sobel_thresh(img_unwarp)
    # Sobel Magnitude (using default parameters)
    # img_sobelMag = mag_thresh(img_unwarp)
    # Sobel Direction (using default parameters)
    # img_sobelDir = dir_thresh(img_unwarp)
    # HLS S-channel Threshold (using default parameters)
    # img_SThresh = hls_sthresh(img_unwarp)
    # HLS L-channel Threshold (using default parameters)
    img_LThresh = hls_lthresh(img_unwarp_crop)
    # Lab B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp_crop)
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_BThresh)
    # combined = np.zeros_like(img_SThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    # combined[(img_LThresh == 1) | (img_SThresh == 1)] = 1
    return img_unwarp_crop, combined, Minv

def ROI_mask(image):
    height = image.shape[0]
    width = image.shape[1]
    # A triangular polygon to segment the lane area and discarded other irrelevant parts in the image
    # Defined by three (x, y) coordinates
    polygons = np.array([
        [(0, height), (round(width / 2), round(height / 2)), (1000, height)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)  ## 255 is the mask color
    # Bitwise AND between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_image(img,mode):
    new_img = np.copy(img)
    img_unwarp_crop, img_bin, Minv = pipeline(new_img)

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, _ = sliding_window_polyfit(img_bin)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)

    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        h = img.shape[0]
        l_fit_x_int = l_fit[0] * h ** 2 + l_fit[1] * h + l_fit[2]
        r_fit_x_int = r_fit[0] * h ** 2 + r_fit[1] * h + r_fit[2]
        x_int_diff = abs(r_fit_x_int - l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)

    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        img_out1 = draw_lane(mode,new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv, img_unwarp_crop)
        rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                               l_lane_inds, r_lane_inds)
        img_out = draw_data(img_out1, (rad_l + rad_r) / 2, d_center)
    else:
        img_out = new_img

    return img_out
