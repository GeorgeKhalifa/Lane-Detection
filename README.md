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

# Advanced Lane Detection

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
![download](https://user-images.githubusercontent.com/74133869/164295170-d0bd3a1e-ed42-4e2a-b433-5719ee2c4af1.png)

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


![download](https://user-images.githubusercontent.com/74133869/164298153-f32f3116-ca93-4913-b378-24b782752723.png)


  
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

![download](https://user-images.githubusercontent.com/74133869/164298342-f7b1e86f-4526-4638-8f23-dd842d612787.png)


```python
def get_hls_lthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls_img[:, :, 1]

    binary_output = np.zeros_like(L).astype(np.uint8)    
    binary_output[(L >= thresh[0]) & (L < thresh[1])] = 1
    
    return binary_output
```


```python
img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_hls_lthresh_img(img, thresh=(201, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
```




    Text(0.5, 1.0, 'Transformed:: test_images/test5.jpg')




![png](output_30_1.png)



```python
def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)    
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1
    
    return binary_output
```


```python
img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_hls_sthresh_img(img, thresh=(150, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
```




    Text(0.5, 1.0, 'Transformed:: test_images/test5.jpg')




![png](output_32_1.png)



```python
def get_lab_athresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab_img[:, :, 1]
    
    bin_op = np.zeros_like(A).astype(np.uint8)
    bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1
    
    return bin_op
```


```python
def get_lab_bthresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]
    
    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1
    
    return bin_op
```


```python
img = undistort(images[0], objpoints, imgpoints)
    
combined_binary = get_lab_bthresh_img(img, thresh=(147, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image , fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: "+ image, fontsize=18)
```




    Text(0.5, 1.0, 'Transformed:: test_images/test5.jpg')




![png](output_35_1.png)



```python
def get_bin_img(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0,255), r_thresh=(0, 255), 
                s_thresh=(0,255), b_thresh=(0, 255), g_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel_size)
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    
    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    r_binary = get_rgb_thresh_img(img, thresh= r_thresh)
    
    # Threshhold G color channel
    g_binary = get_rgb_thresh_img(img, thresh= g_thresh, channel='G')
    
    # Threshhold B in LAB
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)
    
    # Threshold color channel
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1)| (b_binary == 1) | (g_binary == 1)] = 1

    return combined_binary
```


```python
# Testing the threshholding
kernel_size = 5
mag_thresh = (30, 100)
r_thresh = (235, 255)
s_thresh = (165, 255)
b_thresh = (160, 255)
g_thresh = (210, 255)

for image_name in images:
    img = undistort(image_name, objpoints, imgpoints)
    
    combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
                                  s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image , fontsize=18)
    ax2.imshow(combined_binary, cmap='gray')
    ax2.set_title("Threshold Binary:: "+ image, fontsize=18)
    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
```


![png](output_37_0.png)



![png](output_37_1.png)



![png](output_37_2.png)



![png](output_37_3.png)



![png](output_37_4.png)



![png](output_37_5.png)


# Perspective Transform

Perspective transform maps the points in given image to different perspective. <br/>
We are here looking for bird's eye view of the road <br/>
This will be helpful in finding lane curvature. <br/>
Note that after perspective transform the lanes should apear aproximately parallel <br/>


```python
def transform_image(img, offset=250, src=None, dst=None):    
    img_size = (img.shape[1], img.shape[0])
    
    out_img_orig = np.copy(img)
       
    leftupper  = (585, 460)
    rightupper = (705, 460)
    leftlower  = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
    
    
    warped_leftupper = (offset,0)
    warped_rightupper = (offset, img.shape[0])
    warped_leftlower = (img.shape[1] - offset, 0)
    warped_rightlower = (img.shape[1] - offset, img.shape[0])
    
    color_r = [0, 0, 255]
    color_g = [0, 255, 0]
    line_width = 5
    
    if src is not None:
        src = src
    else:
        src = np.float32([leftupper, leftlower, rightupper, rightlower])
        
    if dst is not None:
        dst = dst
    else:
        dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])
    
    cv2.line(out_img_orig, leftlower, leftupper, color_r, line_width)
    cv2.line(out_img_orig, leftlower, rightlower, color_r , line_width * 2)
    cv2.line(out_img_orig, rightupper, rightlower, color_r, line_width)
    cv2.line(out_img_orig, rightupper, leftupper, color_g, line_width)
    
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    out_warped_img = np.copy(warped)
    
    cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_r, line_width)
    cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_r , line_width * 2)
    cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_r, line_width)
    cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_g, line_width)
    
    return warped, M, minv, out_img_orig, out_warped_img
```


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
