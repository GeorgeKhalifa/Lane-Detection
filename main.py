import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import joblib
import matplotlib.image as mpimg
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from car_detection import *
from moviepy.editor import VideoFileClip
from helper_functions import *
import sys


# Press the green button in the gutter to run the script.
vi_in = sys.argv[1]
vi_ou = sys.argv[2]
mlp = joblib.load('mlp1.pkl')
X_scaler = joblib.load('scaler1.pkl')
Boxes = boxes()
video_output1 = vi_ou
video_input1 = VideoFileClip(vi_in)
processed_video = video_input1.fl_image(lambda image: process_vid(image))
processed_video.write_videofile(video_output1, audio=False)



