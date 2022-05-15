import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from model import *
from car_detection import *

mlp = joblib.load('mlp1.pkl')
X_scaler = joblib.load('scaler1.pkl')
vehicles = glob.glob('C://Users//John Emad//Desktop//imageprocessing//Data//vehicles//*//*.png')
non_vehicles = glob.glob('C://Users//John Emad//Desktop//imageprocessing//Data//non-vehicles//*//*.png')

#######extarct features it returns the featues of image using get_feature_space and HOGDescriptor
def extract_features(imgs, cspace='RGB', size = (64,64)):
    features = []
    for filename in imgs:
        image = imread(filename)
        if size != (64,64):
            image = cv2.resize(image, size)
        features.append(
            np.ravel(
                cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9) \
                    .compute(get_feature_space(image, cspace))
            )
        )
    return features

#######getfeature space and check it is converted from RGB into the specified color-space
def get_feature_space(img, cspace):
    if cspace != 'RGB':
        if cspace == 'HLS':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YCrCb':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'HSV':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'YUV':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'Lab':
            features = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        return features
    
#######get hog features uses HOGDescriptor from opencv library
def get_hog_features(img, cspace):
    return np.ravel(
        cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9) \
            .compute(get_feature_space(img, cspace))
    )


#######ploting feature function in it we use the standardscaler() tand randomization to get features value
def plot_features(vehicle_features, non_vehicle_features):   
    vehicle_features[0].shape
    if len(vehicle_features) > 0:
        X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        vehicle_ind = np.random.randint(0, len(vehicles))
        
        ## drawing is commented as it was used in the notebook
        
        # fig = plt.figure(figsize=(12,4))
        # plt.subplot(131)
        # plt.imshow(mpimg.imread(vehicles[vehicle_ind]))
        # plt.title('original image')
        # plt.subplot(132)
        # plt.plot(scaled_X[vehicle_ind])
        # plt.title('scaled features')
        # fig.tight_layout()
    return scaled_X, X_scaler


