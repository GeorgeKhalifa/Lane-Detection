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


def process_vid(image):
    detected_vehicles = []
    pxs = 320
    PXS_LIMIT = 720
    y_start_stop = [400, 660]
    xy_overlap = (0.8, 0.8)
    ACCEPTANCE_THRESHOLD = .98
    INCREMENT_PXS_BY = 16
    while pxs < PXS_LIMIT:
        windows = slide_window(
            image,
            x_start_stop=[640, None],
            y_start_stop=y_start_stop,
            xy_window=(pxs, pxs),
            xy_overlap=xy_overlap
        )
        for w in windows:
            features = []
            resized = cv2.resize(
                (
                    image[w[0][1]: w[1][1], w[0][0]: w[1][0]]
                ), (64, 64)
            )

            hf = get_hog_features(resized, cspace='YUV')
            x_scaled = X_scaler.transform(hf.reshape(1, -1))
            if resized.shape[0] > 0:
                if mlp.predict_proba(x_scaled.reshape(1, -1))[0][1] > ACCEPTANCE_THRESHOLD:
                    detected_vehicles.append(w)
        pxs += INCREMENT_PXS_BY

    out = np.copy(image).astype('uint8')
    boxes = draw_boxes(np.zeros_like(image), bboxes=detected_vehicles, thick=-1)
    rects_arr = []
    contours, _ = cv2.findContours(boxes[:, :, 2].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        rects_arr.append([x, y, x + width, y + height])
    Boxes.detections.append(rects_arr)
    boxes = []
    combined = np.ravel(np.array(Boxes.detections))
    for i in range(len(combined)):
        boxes.extend(np.ravel(combined[i]))
    bb = []
    i = 0
    while i <= len(boxes) - 3:
        bb.append(boxes[i:i + 4])
        i += 4
    rects, _ = cv2.groupRectangles(np.array(bb).tolist(), 10, .1)
    for r in rects:
        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 5)
    return out