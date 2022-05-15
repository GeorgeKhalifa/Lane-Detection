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


def annotate_img(path):
    image = imread(path)
    detected_vehicles = []
    pxs = 320
    INCREMENT_SIZE_BY = 16
    PXS_LIMIT = 720
    y_start_stop = [400, 660]
    xy_overlap = (0.8, 0.8)
    ACCEPTANCE_THRESHOLD = .98

    while pxs < PXS_LIMIT:
        windows = slide_window(
            image,
            x_start_stop=[None, None],
            y_start_stop=y_start_stop,
            xy_window=(pxs, pxs),
            xy_overlap=xy_overlap
        )
        for window in windows:
            features = []
            resized = cv2.resize((image[window[0][1]: window[1][1], window[0][0]: window[1][0]]), (64, 64))
            hog_features = get_hog_features(resized, cspace='YUV')

            x_scaled = X_scaler.transform(hog_features.reshape(1, -1))

            if resized.shape[0] > 0:
                if mlp.predict_proba(x_scaled.reshape(1, -1))[0][1] > ACCEPTANCE_THRESHOLD:
                    detected_vehicles.append(window)
        pxs += INCREMENT_SIZE_BY

    out = np.copy(image)
    boxes = draw_boxes(np.zeros_like(image), bboxes=detected_vehicles, thick=-1)
    contours, _ = cv2.findContours(boxes[:, :, 2].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for countour in contours:
        rect_color_tup = (0, 255, 0)
        x, y, width, height = cv2.boundingRect(countour)
        cv2.rectangle(out, (x, y), (x + width, y + height), rect_color_tup, 6)
        moments = cv2.moments(countour)
        cv2.circle(
            out, (
                int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
            ), 15, (255, 0, 0), -1
        )
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
    f.tight_layout()
    ax1.axis('on')
    ax1.set_title('original')
    ax1.imshow(image)
    ax2.axis('on')
    ax2.set_title('car detected box spots')
    ax2.imshow(boxes, cmap='hot')
    ax3.axis('on')
    ax3.set_title('Annotated car')
    ax3.imshow(out)