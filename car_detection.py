from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *
from helper_functions import get_hog_features
from model import *

mlp = joblib.load('mlp1.pkl')
X_scaler = joblib.load('scaler1.pkl')

class boxes:
    def __init__(self):
        self.detections = deque(maxlen=12)

Boxes = boxes()
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    out_img = np.copy(img)
    for b in bboxes:
        cv2.rectangle(out_img, *b, color, thick)
    return out_img

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0] ### if both are NONE xspan=img.shape[1]  width=1280px
    yspan = y_start_stop[1] - y_start_stop[0] ### if both are NONE yspan=img.shape[0]  width=720px
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))  ### 64*(1-0.75)=16
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))  ### 64*(1-0.75)=16
    nx_windows = np.int(xspan/nx_pix_per_step) ###1280/16=80
    ny_windows = np.int(yspan/ny_pix_per_step) ###260/16=16
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = (xs+1)*nx_pix_per_step + x_start_stop[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = (ys+1)*ny_pix_per_step + y_start_stop[0]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list