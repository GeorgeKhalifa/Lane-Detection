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
import line
from pipeline import *

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

# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # print(histogram.shape)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)  # 640
    quarter_point = np.int(midpoint // 2)  # 320
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point  ####(450+320)
    rightx_base = np.argmax(histogram[midpoint:(midpoint + quarter_point)]) + midpoint  ##(840+640)

    # print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height  # 720-(1)*72
        win_y_high = img.shape[0] - window * window_height  # 720-(0)*72
        win_xleft_low = leftx_current - margin  # 471-80
        win_xleft_high = leftx_current + margin  # 471+80
        win_xright_low = rightx_current - margin  # 851-80
        win_xright_high = rightx_current + margin  # 851+80
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data

# Define method to draw slidings and lines
def draw_sliding(exampleImg_bin):
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)
    h = exampleImg_bin.shape[0]
    if left_fit is not None and right_fit is not None:
        left_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
        right_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
    # print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

    rectangles = visualization_data[0]
    histogram = visualization_data[1]

    # Create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin)) * 255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, exampleImg_bin.shape[0] - 1, exampleImg_bin.shape[0])
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
        # Identify the x and y positions of all nonzero pixels in the image
    nonzero = exampleImg_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
    # plt.imshow(out_img)

    # temp_x = np.zeros_like(left_fitx)
    # for x in left_fitx:
    # x = x / 1280 * 200
    if left_fit is not None and right_fit is not None:
        r_left = np.column_stack((left_fitx, ploty))
        r_right = np.column_stack((right_fitx, ploty))
        # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        r_left = r_left.reshape((-1, 1, 2))
        r_right = r_right.reshape((-1, 1, 2))
        out_img = cv2.polylines(out_img, np.int32([r_left]), False, (255, 255, 0), 5)
        out_img = cv2.polylines(out_img, np.int32([r_right]), False, (255, 255, 0), 5)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.imshow(out_img)
    return out_img

# Define method to fit polynomial to binary image based upon a previous fit (chronologically speaking);
# this assumes that the fit will not change significantly from one video frame to the next
def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (
                left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[2] - margin)) &
                      (nonzerox < (left_fit_prev[0] * (nonzeroy ** 2) + left_fit_prev[1] * nonzeroy + left_fit_prev[
                          2] + margin)))
    right_lane_inds = ((nonzerox > (
                right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[2] - margin)) &
                       (nonzerox < (right_fit_prev[0] * (nonzeroy ** 2) + right_fit_prev[1] * nonzeroy + right_fit_prev[
                           2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds

#Write the title of each stage of the pipline on the output video
def write_titles(original_img, title):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(new_img, title, (10,15), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
    direction = ''
    return new_img

#Draw the pipeline stages on the output video
def combine_images(color_warp, original_img, result, img_unwarp_crop, binary_img):
    """
    Returns a new image made up of the lane area image, and the remaining lane images are overlaid as
    small images in a row at the top of the the new image
    """
    h, w = original_img.shape[:2]
    src = np.float32([(575, 464),
                      (707, 464),
                      (1049, 682),
                      (258, 682)])
    dst = np.float32([(450, 0),
                      (w - 450, 0),
                      (w - 450, h),
                      (450, h)])

    pts = src.reshape((-1, 1, 2))

    isClosed = True

    # Blue color in BGR
    color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 2

    new_original_img = np.copy(original_img)
    image = cv2.polylines(new_original_img, np.int32([pts]), isClosed, color, thickness)

    sliding_image = draw_sliding(binary_img)

    small_img_unwarp_crop = cv2.resize(img_unwarp_crop, (200, 200))
    small_binary_img = cv2.resize(binary_img, (200, 200))
    small_original_img = cv2.resize(image, (200, 200))
    small_sliding_image = cv2.resize(sliding_image, (200, 200))  # ok
    small_color_warp = cv2.resize(color_warp, (200, 200))  # ok

    img2 = np.zeros_like(small_img_unwarp_crop)
    img2[:, :, 0] = small_binary_img * 255
    img2[:, :, 1] = small_binary_img * 255
    img2[:, :, 2] = small_binary_img * 255

    img2 = write_titles(img2, 'Birdeye view threshold')
    small_color_warp = write_titles(small_color_warp, 'Polynomial fit')
    small_sliding_image = write_titles(small_sliding_image, 'sliding window result')
    small_img_unwarp_crop = write_titles(small_img_unwarp_crop, 'Birdeye view')

    small_img_x_offset = 20
    small_img_size = (200, 200)
    small_img_y_offset = 10
    img_dimensions = (720, 1280)
    result[small_img_y_offset: small_img_y_offset + small_img_size[1],
    600: 600 + small_img_size[0]] = small_img_unwarp_crop

    start_offset_y = small_img_y_offset
    start_offset_x = 2 * small_img_x_offset + small_img_size[0] + 580
    result[start_offset_y: start_offset_y + small_img_size[1],
    start_offset_x: start_offset_x + small_img_size[0]] = img2

    start_offset_y = small_img_y_offset
    start_offset_x = 3 * small_img_x_offset + 2 * small_img_size[0] + 580
    result[start_offset_y: start_offset_y + small_img_size[1],
    start_offset_x: start_offset_x + small_img_size[0]] = small_original_img

    start_offset_y = small_img_y_offset * 2 + small_img_size[1]  # 2+10 + 200 = 220 --> 420
    start_offset_x = 3 * small_img_x_offset + 2 * small_img_size[0] + 580  # 3*20 + 2*200 + 580 = 1040 --> 1240
    result[start_offset_y: start_offset_y + small_img_size[1],
    start_offset_x: start_offset_x + small_img_size[0]] = small_sliding_image

    start_offset_y = small_img_y_offset * 2 + small_img_size[1]
    start_offset_x = 3 * small_img_x_offset + small_img_size[0] + 560
    result[start_offset_y: start_offset_y + small_img_size[1],
    start_offset_x: start_offset_x + small_img_size[0]] = small_color_warp

    return result

#Draw the center place and the curvature on the video
def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1, (200,255,155), 2, cv2.LINE_AA)
    return new_img

