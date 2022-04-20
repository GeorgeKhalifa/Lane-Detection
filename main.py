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
from line import *
from pipeline import *
import helper_functions
import sys




    #'test_video_challenge_1.mp4'
    # 'challenge_video.mp4'

#    take_input(sys.argv[1], sys.argv[2], sys.argv[3])

vi_mood = int(sys.argv[1])
vi_in = sys.argv[2]
vi_ou = sys.argv[3]
video_output1 = vi_ou
video_input1 = VideoFileClip(vi_in)  # .subclip(22,26)
print(vi_mood)
processed_video = video_input1.fl_image(lambda image: process_image(image, vi_mood))
processed_video.write_videofile(video_output1, audio=False)
