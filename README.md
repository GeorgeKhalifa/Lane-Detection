# Lane-Detection
</br>
</br>
<h1>Libraries needed to be installed before running the code</h1>
</br>
<ul>
  <li><span>numpy :</span>  pip install numpy</li>
  <li><span>OpenCV :</span> pip install opencv-contrib-python</li>
  <li><span>matpoltlib :</span> pip install matpoltlib</li>
  <li><span>moviepy :</span> pip install moviepy</li>
</ul>
</br>

## Pipeline of this project:

1) Compute the camera caliberation matrix and distortion coefficient from chessboard images. <br/>
2) Apply distortion correction to raw images. <br/>
3) Use color gradient to create binary threshholded image. <br/>
4) Apply perspective transform to binary threshholded image to get top view. <br/>
5) Detect pixel lane and fit to find lane boundary. <br/>
6) Determine lane curvature and vehicle position wrt centre. <br/>
7) Warp the detected boundaries back to original image. <br/>


```python
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

%matplotlib inline
```
```python
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
```
# Read Image

Read input image from a path then convert it from BGR to RGB and plot it <br/>

```python
# Choose an image from which to build and demonstrate each step of the pipeline
exampleImg = cv2.imread('./test_images/test6.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
plt.imshow(exampleImg)
```
![out1](https://user-images.githubusercontent.com/74133869/164293758-c5955f09-0f16-45ca-991a-3c6555edd9f1.png)

# UNwarp (hawkeye)

```python
def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv
```

```python
h,w = exampleImg.shape[:2]

# define source and destination points for transform

src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

exampleImg_unwarp, M, Minv = unwarp(exampleImg, src, dst)

# Visualize unwarp
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
f.subplots_adjust(hspace = .2, wspace=.05)
ax1.imshow(exampleImg)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax1.plot(x, y, color='#33cc99', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
ax1.set_ylim([h,0])
ax1.set_xlim([0,w])
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(exampleImg_unwarp)
ax2.set_title('Unwarped Image', fontsize=30)
```
![download](https://user-images.githubusercontent.com/74133869/164294681-347ea621-91e2-43c6-823c-f0353bad1bab.png)

# channels for testing

```python
# Visualize multiple color space channels
exampleImg_unwarp_R = exampleImg_unwarp[:,:,0]
exampleImg_unwarp_G = exampleImg_unwarp[:,:,1]
exampleImg_unwarp_B = exampleImg_unwarp[:,:,2]
exampleImg_unwarp_HSV = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HSV)
exampleImg_unwarp_H = exampleImg_unwarp_HSV[:,:,0]
exampleImg_unwarp_S = exampleImg_unwarp_HSV[:,:,1]
exampleImg_unwarp_V = exampleImg_unwarp_HSV[:,:,2]
exampleImg_unwarp_LAB = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2Lab)
exampleImg_unwarp_L = exampleImg_unwarp_LAB[:,:,0]
exampleImg_unwarp_A = exampleImg_unwarp_LAB[:,:,1]
exampleImg_unwarp_B2 = exampleImg_unwarp_LAB[:,:,2]
exampleImg_unwarp_HLS = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HLS)
exampleImg_unwarp_H22 = exampleImg_unwarp_HLS[:,:,0]
exampleImg_unwarp_L22 = exampleImg_unwarp_HLS[:,:,1]
exampleImg_unwarp_S22 = exampleImg_unwarp_HLS[:,:,2]

fig, axs = plt.subplots(4,3, figsize=(16, 12))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
axs[0].imshow(exampleImg_unwarp_R, cmap='gray')
axs[0].set_title('RGB R-channel', fontsize=30)
axs[1].imshow(exampleImg_unwarp_G, cmap='gray')
axs[1].set_title('RGB G-Channel', fontsize=30)
axs[2].imshow(exampleImg_unwarp_B, cmap='gray')
axs[2].set_title('RGB B-channel', fontsize=30)
axs[3].imshow(exampleImg_unwarp_H, cmap='gray')
axs[3].set_title('HSV H-Channel', fontsize=30)
axs[4].imshow(exampleImg_unwarp_S, cmap='gray')
axs[4].set_title('HSV S-channel', fontsize=30)
axs[5].imshow(exampleImg_unwarp_V, cmap='gray')
axs[5].set_title('HSV V-Channel', fontsize=30)

axs[6].imshow(exampleImg_unwarp_L, cmap='gray')
axs[6].set_title('LAB L-channel', fontsize=30)
axs[7].imshow(exampleImg_unwarp_A, cmap='gray')
axs[7].set_title('LAB A-Channel', fontsize=30)
axs[8].imshow(exampleImg_unwarp_B2, cmap='gray')
axs[8].set_title('LAB B-Channel', fontsize=30)

axs[9].imshow(exampleImg_unwarp_H22, cmap='gray')
axs[9].set_title('HLS H-channel', fontsize=30)
axs[10].imshow(exampleImg_unwarp_L22, cmap='gray')
axs[10].set_title('HLS L-Channel', fontsize=30)
axs[11].imshow(exampleImg_unwarp_S22, cmap='gray')
axs[11].set_title('HLS S-Channel', fontsize=30)
```
![download](https://user-images.githubusercontent.com/74133869/164294947-8509ec9f-3492-4bce-b236-5d9051288d89.png)

# Sobel filter for edge detection

```python
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale === or LAB L channel
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2HLS))[:,:,1] ################ change the channel from L to another
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    #######sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient=='x'), int(orient=='y'))
    #plt.imshow(sobel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary # Remove this line
    return binary_output
```


```python
def update(min_thresh, max_thresh):
    exampleImg_sobelAbs = abs_sobel_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)
    # Visualize sobel absolute threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelAbs, cmap='gray')
    ax2.set_title('Sobel Absolute', fontsize=30)

interact(update, 
         min_thresh=(0,255), 
         max_thresh=(0,255))

```
![download](https://user-images.githubusercontent.com/74133869/164300817-3986b8e6-2ba1-4963-89b6-6b1d0e592a02.png)

```python
# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    blur= cv2.GaussianBlur(gray,(13,13),0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude 
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sxbinary) 
    return binary_output
```

```python
def update(kernel_size, min_thresh, max_thresh):
    exampleImg_sobelMag = mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh, max_thresh))
    # Visualize sobel magnitude threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelMag, cmap='gray')
    ax2.set_title('Sobel Magnitude', fontsize=30)

interact(update, kernel_size=(1,31,2), 
                 min_thresh=(0,255), 
                 max_thresh=(0,255))
```


![download](https://user-images.githubusercontent.com/74133869/164300976-c51c305f-728d-4c70-935d-a38a364b541e.png)



  
```python
# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel=7, thresh=(0, 0.09)):    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
```

```python
def update(kernel_size, min_thresh, max_thresh):
    exampleImg_sobelDir = dir_thresh(exampleImg_unwarp, kernel_size, (min_thresh, max_thresh))
    # Visualize sobel direction threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_sobelDir, cmap='gray')
    ax2.set_title('Sobel Direction', fontsize=30)

interact(update, kernel_size=(1,31,2), 
                 min_thresh=(0,np.pi/2,0.01), 
                 max_thresh=(0,np.pi/2,0.01))
```

![download](https://user-images.githubusercontent.com/74133869/164301378-066b0fc2-675a-464d-8bfc-48a98d6e1c2d.png)



```python
def update(mag_kernel_size, mag_min_thresh, mag_max_thresh, dir_kernel_size, dir_min_thresh, dir_max_thresh):
    exampleImg_sobelMag2 = mag_thresh(exampleImg_unwarp, mag_kernel_size, (mag_min_thresh, mag_max_thresh))
    exampleImg_sobelDir2 = dir_thresh(exampleImg_unwarp, dir_kernel_size, (dir_min_thresh, dir_max_thresh))
    combined = np.zeros_like(exampleImg_sobelMag2)
    combined[((exampleImg_sobelMag2 == 1) & (exampleImg_sobelDir2 == 1))] = 1
    # Visualize sobel magnitude + direction threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Sobel Magnitude + Direction', fontsize=30)

interact(update, mag_kernel_size=(1,31,2), 
                 mag_min_thresh=(0,255), 
                 mag_max_thresh=(0,255),
                 dir_kernel_size=(1,31,2), 
                 dir_min_thresh=(0,np.pi/2,0.01), 
                 dir_max_thresh=(0,np.pi/2,0.01))
```



![download](https://user-images.githubusercontent.com/74133869/164301568-98a8fd43-aa29-45a0-b581-0497fa248b44.png)

# Thresholding for S Channel HLS

```python
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_sthresh(img, thresh=(125, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = cv2.GaussianBlur(hls,(9,9),0)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```


```python
def update(min_thresh, max_thresh):
    exampleImg_SThresh = hls_sthresh(exampleImg_unwarp, (min_thresh, max_thresh))
    # Visualize hls s-channel threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_SThresh, cmap='gray')
    ax2.set_title('HLS S-Channel', fontsize=30)

interact(update,
         min_thresh=(0,255), 
         max_thresh=(0,255))
```


![download](https://user-images.githubusercontent.com/74133869/164302040-fa8a6293-4999-4cc9-a88b-dcf01ff90b9f.png)



```python
# Define a function that thresholds the L-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_lthresh(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls = cv2.GaussianBlur(hls,(25,25),0)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
```


```python
def update(min_thresh, max_thresh):
    exampleImg_LThresh = hls_lthresh(exampleImg_unwarp, (min_thresh, max_thresh))
    # Visualize hls l-channel threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    l_channel=exampleImg_unwarp[:,:,1]
    hls = cv2.GaussianBlur(l_channel,(51,51),0)
    ax1.imshow(hls, cmap='gray')
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_LThresh, cmap='gray')
    ax2.set_title('HLS L-Channel', fontsize=30)

interact(update,
         min_thresh=(0,255), 
         max_thresh=(0,255))
```

![download](https://user-images.githubusercontent.com/74133869/164302207-cf1c26df-932c-4f34-8a1a-d7f4a6b5c62b.png)


# Thresholding for B Channel LAB

```python
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
```


```python
def update(min_b_thresh, max_b_thresh):
    exampleImg_LBThresh = lab_bthresh(exampleImg_unwarp, (min_b_thresh, max_b_thresh))
    # Visualize LAB B threshold
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(exampleImg_unwarp)
    ax1.set_title('Unwarped Image', fontsize=30)
    ax2.imshow(exampleImg_LBThresh, cmap='gray')
    ax2.set_title('LAB B-channel', fontsize=30)

interact(update,
         min_b_thresh=(0,255),
         max_b_thresh=(0,255))
```
![download](https://user-images.githubusercontent.com/74133869/164302450-324046ac-0ef4-406d-b550-764438850f75.png)


# Pipeline of Lane Detection

```python
# Define the complete image processing pipeline, reads raw image and returns binary image with lane lines identified
# (hopefully)
def pipeline(img):
    #img_unwarp=canny_edge_detector(img)
    #img_unwarp_crop=ROI_mask(img_unwarp)
    # Perspective Transform
    img_unwarp_crop, M, Minv = unwarp(img, src, dst)

    # Sobel Absolute (using default parameters)
    #img_sobelAbs = abs_sobel_thresh(img_unwarp)

    # Sobel Magnitude (using default parameters)
    #img_sobelMag = mag_thresh(img_unwarp)
    
    # Sobel Direction (using default parameters)
    #img_sobelDir = dir_thresh(img_unwarp)
    
    # HLS S-channel Threshold (using default parameters)
    #img_SThresh = hls_sthresh(img_unwarp)

    # HLS L-channel Threshold (using default parameters)
    img_LThresh = hls_lthresh(img_unwarp_crop)

    # Lab B-channel Threshold (using default parameters)
    img_BThresh = lab_bthresh(img_unwarp_crop)
    
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_BThresh)
    #combined = np.zeros_like(img_SThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    #combined[(img_LThresh == 1) | (img_SThresh == 1)] = 1
    return img_unwarp_crop,combined, Minv
```

```python
# Make a list of example images
images = glob.glob('./test_images/*.jpg')
                                          
# Set up plot
fig, axs = plt.subplots(len(images),3, figsize=(10, 20))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
                  
i = 0
for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_unwarp,img_bin, Minv = pipeline(img)
    axs[i].imshow(img)
    axs[i].axis('off')
    i += 1
    axs[i].imshow(img_bin, cmap='gray')
    i+=1
    axs[i].imshow(img_unwarp, cmap='gray')
    axs[i].axis('off')
    i += 1
```
![download](https://user-images.githubusercontent.com/74133869/164302713-7bb00360-b3cf-49a5-a481-409b4f374e90.png)


# Helper functions

```python
# Define method to fit polynomial to binary image with lines extracted, using sliding window
def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    #print(histogram.shape)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2) # 640
    quarter_point = np.int(midpoint//2)      # 320
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point  ####(450+320)
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint  ##(840+640)
    
    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
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
        win_y_low = img.shape[0] - (window+1)*window_height    #720-(1)*72
        win_y_high = img.shape[0] - window*window_height       #720-(0)*72
        win_xleft_low = leftx_current - margin                 #471-80
        win_xleft_high = leftx_current + margin                #471+80
        win_xright_low = rightx_current - margin               #851-80
        win_xright_high = rightx_current + margin              #851+80
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
```

```python
# visualize the result on example image
exampleImg = cv2.imread('./test_images/test7.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
# plt.imshow(exampleImg)
img_unwarp_crop,exampleImg_bin, Minv = pipeline(exampleImg)
print()
# plt.imshow(img_unwarp_crop)
# plt.imshow(exampleImg_bin)
# print(Minv)

def draw_sliding(exampleImg_bin):
    
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)
    h = exampleImg.shape[0]
    if left_fit is not None and right_fit is not None:
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    #print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)
    
    rectangles = visualization_data[0]
    histogram = visualization_data[1]



    # Create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin))*255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, exampleImg_bin.shape[0]-1, exampleImg_bin.shape[0] )
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)
        # Identify the x and y positions of all nonzero pixels in the image
    nonzero = exampleImg_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
    #plt.imshow(out_img)

    # temp_x = np.zeros_like(left_fitx)
    # for x in left_fitx:
    # x = x / 1280 * 200
    if left_fit is not None and right_fit is not None:
        r_left=np.column_stack((left_fitx,ploty))
        r_right=np.column_stack((right_fitx,ploty))
    #pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        r_left= r_left.reshape((-1,1,2))
        r_right= r_right.reshape((-1,1,2))
        out_img = cv2.polylines(out_img,np.int32([r_left]),False,(255,255,0),5)
        out_img = cv2.polylines(out_img,np.int32([r_right]),False,(255,255,0),5)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.imshow(out_img)
    return out_img        
    
test_image = draw_sliding(exampleImg_bin)
# plt.imshow(cv2.resize(test_image,(300, 300)))
#plt.title('test')

```

```python
# Print histogram from sliding window polyfit for example image
left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(exampleImg_bin)
histogram = visualization_data[1]
plt.plot(histogram)
plt.xlim(0, 1280)

```

![download](https://user-images.githubusercontent.com/74133869/164303097-fd3d810b-704a-45f1-9e40-c8d008b76820.png)


# Perspective Transform

Perspective transform maps the points in given image to different perspective. <br/>
We are here looking for bird's eye view of the road <br/>
This will be helpful in finding lane curvature. <br/>
Note that after perspective transform the lanes should apear aproximately parallel <br/>





```python
for image in images:
    img = cv2.imread(image)
    warped, M, minv, out_img_orig, out_warped_img = transform_image(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(out_img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(cv2.cvtColor(out_warped_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
```


![png](output_40_0.png)



![png](output_40_1.png)



![png](output_40_2.png)



![png](output_40_3.png)



![png](output_40_4.png)



![png](output_40_5.png)



```python
for image in images:
    img = undistort(image, objpoints, imgpoints)
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, 
                                  r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title("Transformed:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
```


![png](output_41_0.png)



![png](output_41_1.png)



![png](output_41_2.png)



![png](output_41_3.png)



![png](output_41_4.png)



![png](output_41_5.png)


# Lane line pixel detection and polynomial fitting

With binary image where lane lines are clearly visible, now we have to decide lane pixels <br/>
Also we need to decide pixels from left lane and pixels from right lane. <br/>
<br/>
The threshold image pixels are either 0 or 1, so if we take histogram of the image <br/>
the 2 peaks that we might see in histogram might be good position to start to find lane pixels <br/>
We can then use sliding window to find further pixels<br/>


```python
def find_lines(warped_img, nwindows=9, margin=80, minpix=40):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low =  rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50, show=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
        = find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    if show == True:
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img
```

# Skip the sliding windows step once you've found the lines

Once lines are found, we don't need to do blind search , but we can search around existing line with some margin. As the lanes are not going to shift much between 2 frames of video


```python
def search_around_poly(binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit second order polynomial to for for points on real world   
    left_lane_indices = np.polyfit(lefty*ymtr_per_pixel, leftx*xmtr_per_pixel, 2)
    right_lane_indices = np.polyfit(righty*ymtr_per_pixel, rightx*xmtr_per_pixel, 2)
    
    return left_fit, right_fit, left_lane_indices, right_lane_indices
```


```python
left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=20)
plt.imshow(out_img)
plt.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
```


![png](output_46_0.png)


# Radius of curvature

We can fit the circle that can approximately fits the nearby points locally <br/>

![alt text](radius_curvature1.png)

The radius of curvature is radius of the circle that fits the curve<br/>
The radius of curvature can be found out using equation: <br/>
<br/>
![alt text](eq1.gif)
<br/>
For polynomial below are the equation: <br/>
![alt text](eq2.gif)


```python
def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    
    # find radii of curvature
    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_rad, right_rad)
```


```python
def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ## Image mid horizontal position 
    #xmax = img.shape[1]*xmtr_per_pixel
    ymax = img.shape[0]*ymtr_per_pixel
    
    center = img.shape[1] / 2
    
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))
    
    return message
```


```python
def draw_lines(img, left_fit, right_fit, minv):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Find left and right points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix 
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)
```


```python
def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt = dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    
    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return out_img
```


```python
for image in images:    
    img = undistort(image, objpoints, imgpoints)
    
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
                                  s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    
    xmtr_per_pixel=3.7/800
    ymtr_per_pixel=30/720
    
    left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=12, show=False)
    lane_img = draw_lines(img, left_fit, right_fit, minv)
    out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Lane:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
```


![png](output_52_0.png)



![png](output_52_1.png)



![png](output_52_2.png)



![png](output_52_3.png)



![png](output_52_4.png)



![png](output_52_5.png)


# Pipeline for video


```python
class Lane():
    def __init__(self, max_counter):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left = [np.array([False])] 
        self.current_fit_right=None
        self.best_fit_right = None
        self.history_right = [np.array([False])] 
        self.counter = 0
        self.max_counter = 1
        self.src = None
        self.dst = None
        
    def set_presp_indices(self, src, dest):
        self.src = src
        self.dst = dst
        
    def reset(self):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left =[np.array([False])] 
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right =[np.array([False])] 
        self.counter = 0
        
    def update_fit(self, left_fit, right_fit):
        if self.counter > self.max_counter:
            self.reset()
        else:
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            self.history_left.append(left_fit)
            self.history_right.append(right_fit)
            self.history_left = self.history_left[-self.max_counter:] if len(self.history_left) > self.max_counter else self.history_left
            self.history_right = self.history_right[-self.max_counter:] if len(self.history_right) > self.max_counter else self.history_right
            self.best_fit_left = np.mean(self.history_left, axis=0)
            self.best_fit_right = np.mean(self.history_right, axis=0)
        
    def process_image(self, image):
        img = undistort_no_read(image, objpoints, imgpoints)
        
        combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                      r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    
        if self.src is not None or self.dst is not None:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, src=self.src, dst= self.dst)
        else:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    
        xmtr_per_pixel=3.7/800
        ymtr_per_pixel=30/720
    
        if self.best_fit_left is None and self.best_fit_right is None:
            left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=15, show=False)
        else:
            left_fit, right_fit, left_lane_indices, right_lane_indices= search_around_poly(warped, self.best_fit_left, self.best_fit_right, xmtr_per_pixel, ymtr_per_pixel)
            
        self.counter += 1
        
        lane_img = draw_lines(img, left_fit, right_fit, unwarp_matrix)
        out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
        
        self.update_fit(left_fit, right_fit)
        
        return out_img
```


```python
clip1 = VideoFileClip("project_video.mp4")
img = clip1.get_frame(0)

leftupper  = (585, 460)
rightupper = (705, 460)
leftlower  = (210, img.shape[0])
rightlower = (1080, img.shape[0])
    
color_r = [255, 0, 0]
color_g = [0, 255, 0]
line_width = 5
    
src = np.float32([leftupper, leftlower, rightupper, rightlower])

cv2.line(img, leftlower, leftupper, color_r, line_width)
cv2.line(img, leftlower, rightlower, color_r , line_width * 2)
cv2.line(img, rightupper, rightlower, color_r, line_width)
cv2.line(img, rightupper, leftupper, color_g, line_width)

plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f7e43446358>




![png](output_55_1.png)



```python
lane1 = Lane(max_counter=5)

leftupper  = (585, 460)
rightupper = (705, 460)
leftlower  = (210, img.shape[0])
rightlower = (1080, img.shape[0])
    
warped_leftupper = (250,0)
warped_rightupper = (250, img.shape[0])
warped_leftlower = (1050, 0)
warped_rightlower = (1050, img.shape[0])

src = np.float32([leftupper, leftlower, rightupper, rightlower])
dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

lane1.set_presp_indices(src, dst)

output = "test_videos_output/project.mp4"
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(lane1.process_image)
%time white_clip.write_videofile(output, audio=False)
```

    t:   0%|          | 0/1260 [00:00<?, ?it/s, now=None]

    Moviepy - Building video test_videos_output/project.mp4.
    Moviepy - Writing video test_videos_output/project.mp4
    


                                                                    

    Moviepy - Done !
    Moviepy - video ready test_videos_output/project.mp4
    CPU times: user 39min 39s, sys: 6.49 s, total: 39min 45s
    Wall time: 33min 28s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/project.mp4">
</video>




## 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?




While testing on challenge video and harder challenge video, the problems encountered are mostly due to lighting condition, shadows and road conditions ( edge visible on road other than lane marking). Although HLS space works well for simple video, it activates noisy areas more. ( that's visible in video 2 and 3). I may try LAB color space which separates yellow color better.<br/>
<br/>
The averaging of lane works well to smoothen the polynomial output. Harder challenge also poses a problem with very steep curves too. May be we need to fit higher polynomial to these steep curves<br/>
<br/>
Also, still these algorithms relie a lot on lane being visible, video being taken from certain angle, light condition and still feels like hand crafted. There might be better way based off RNN(ReNet) or Instance Segmentation (https://arxiv.org/pdf/1802.05591.pdf) or spatial CNN (https://arxiv.org/pdf/1712.06080.pdf) <br/>

I wonder what the tesla is using  that they displayed in recent autonomy day video.
<br/>


```python

```
