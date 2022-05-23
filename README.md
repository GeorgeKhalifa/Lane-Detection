# Car-Detection
</br>
<span>Repo link : </span><a href="https://github.com/GeorgeKhalifa/Lane-Detection.git">https://github.com/GeorgeKhalifa/Lane-Detection.git</a>
<h1> Participants Names:</br></h1>
<ul>
  <li><span>John Emad Marcos Sadek :</span>  1700410 </li>
  <li><span>Botros Kamal Botros Khella :</span>  1700369</li>
  <li><span>George Wageeh Shamshoon Khalifa :</span>  1700404</li>
  <li><span>Aya-T-Allah Abdelrehim :</span>  1700337</li>
</ul>
</br>

<h1>Libraries needed to be installed before running the code</h1>
</br>
<ul>
  <li><span>numpy :</span>  pip install numpy</li>
  <li><span>OpenCV :</span> pip install opencv-contrib-python</li>
  <li><span>matpoltlib :</span> pip install matpoltlib</li>
  <li><span>moviepy :</span> pip install moviepy</li>
  <li><span>joblib :</span> pip install joblib</li>
  <li><span>pandas :</span> pip install pandas</li>
  <li><span>sklearn :</span> pip install sklearn</li>
  <li><span>scikit-image :</span> pip install scikit-image --user</li>
</ul>
</br>
<h1>How to run our code</h1></br>
1) Open then edit the open.sh file. </br>
2) Choose the arguments (var1: input video).</br>
3) Choose the arguments (var2: output video).</br>
4) Save the open.sh file.</br>
5) Run the open.sh file.</br>

</br>


## Steps of the project:

1) Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier and Multi-Layer Perceptron MLP Classifier. <br/>
2) Implement a sliding-window technique and use MLP classifier to search for vehicles in images. <br/>
3) Run pipeline on a video stream and reject outliers and follow detected vehicles. <br/>
4) Determine lane curvature and vehicle position wrt centre. <br/>

### 1. Histogram of Oriented Gradients (HOG)
## Steps of HOGDescriptor:
1) preprocessing of image.
2) Calculating Gradients (direction x and y).
3) Calculate the Magnitude and Orientation.
4) Calculate Histogram of Gradients in 8×8 cells (9×1).
5) Normalize gradients in 16×16 cell.
6) Extract the Faetures vector. 

```python
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

def get_hog_features(img, cspace):
    return np.ravel(
        cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9) \
            .compute(get_feature_space(img, cspace))
    )
```
### 2. Final choice of HOG parameters.

First, We defined a function extract_features get_hog_features. This function loops through all images, and creates an array of hogs features of each image. This array is then used as the feature array for training.  Here's a code snippet:
```python
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
```


Of all color spaces, YUV was the best at detecting vehicles. 
We normalized and split by data into train and test sets.
### 3. Training a classifier using selected HOG features.

We trained using both an SVM and an MLP. MLP had a higher test accuracy. Here are the results. 

|Classifier|Training Accuracy|Test Accuracy|
|----------|-----------------|-------------|
|svm |1.00|.956081081081081|
|mlp |1.00|.9926801801801802|

### Sliding Window Search

### 1. Sliding window search, scales, and overlaps. 

We did a bit of research to look for and modify an efficient and accurate sliding window algorithm.

1. get HOGS features for each window
2. only search for vehicle in the bottom half of image
3. multiple window scaled, to ensure we detect both closeby and distant images. 
4. 80% xy overlap, through trial and error

```python
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    nx_windows = np.int(xspan/nx_pix_per_step) 
    ny_windows = np.int(yspan/ny_pix_per_step)
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = (xs+1)*nx_pix_per_step + x_start_stop[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = (ys+1)*ny_pix_per_step + y_start_stop[0]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list
```
