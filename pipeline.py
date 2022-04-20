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



