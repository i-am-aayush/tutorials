## OpenCV Python Tutorial

This tutorial covers essential OpenCV concepts and techniques using Python. Each section explains a concept briefly and provides example code with detailed comments to illustrate the usage. A basic understanding of Python is assumed.

---

### Table of Contents

1. [Introduction to OpenCV](#introduction-to-opencv)
2. [Read, Write, and Display Images](#read-write-and-display-images)
3. [Read, Write, and Display Video from Camera](#read-write-and-display-video-from-camera)
4. [Draw Geometric Shapes on Images](#draw-geometric-shapes-on-images)
5. [Setting Camera Parameters in OpenCV](#setting-camera-parameters-in-opencv)
6. [Show Date and Time on Videos](#show-date-and-time-on-videos)
7. [Handle Mouse Events in OpenCV](#handle-mouse-events-in-opencv)
8. [More Mouse Event Examples](#more-mouse-event-examples)
9. [Splitting and Merging Channels, Resizing, ROI, and Image Arithmetic](#splitting-and-merging-channels-resizing-roi-and-image-arithmetic)
10. [Bitwise Operations (AND, OR, XOR, NOT)](#bitwise-operations-and-or-xor-not)
11. [Binding Trackbars to OpenCV Windows](#binding-trackbars-to-opencv-windows)
12. [Object Detection and Tracking Using HSV Color Space](#object-detection-and-tracking-using-hsv-color-space)
13. [Simple Image Thresholding](#simple-image-thresholding)
14. [Adaptive Thresholding](#adaptive-thresholding)
15. [Matplotlib with OpenCV](#matplotlib-with-opencv)
16. [Morphological Transformations](#morphological-transformations)
17. [Smoothing/Blurring Images](#smoothingblurring-images)
18. [Image Gradients and Edge Detection](#image-gradients-and-edge-detection)
19. [Canny Edge Detection](#canny-edge-detection)
20. [Image Pyramids](#image-pyramids)
21. [Image Blending using Pyramids](#image-blending-using-pyramids)
22. [Find and Draw Contours](#find-and-draw-contours)
23. [Motion Detection and Tracking Using Contours](#motion-detection-and-tracking-using-contours)
24. [Detect Simple Geometric Shapes](#detect-simple-geometric-shapes)
25. [Understanding Image Histograms](#understanding-image-histograms)
26. [Template Matching using OpenCV](#template-matching-using-opencv)
27. [Hough Line Transform Theory](#hough-line-transform-theory)
28. [Hough Line Transform using HoughLines](#hough-line-transform-using-houghlines)
29. [Probabilistic Hough Line Transform using HoughLinesP](#probabilistic-hough-line-transform-using-houghlinesp)
30. [Road Lane Line Detection (Part 1)](#road-lane-line-detection-part-1)
31. [Road Lane Line Detection (Part 2)](#road-lane-line-detection-part-2)
32. [Road Lane Line Detection (Part 3)](#road-lane-line-detection-part-3)
33. [Circle Detection using Hough Circle Transform](#circle-detection-using-hough-circle-transform)
34. [Face Detection using Haar Cascade Classifiers](#face-detection-using-haar-cascade-classifiers)
35. [Eye Detection using Haar Cascade Classifiers](#eye-detection-using-haar-cascade-classifiers)
36. [Detect Corners with Harris Corner Detector](#detect-corners-with-harris-corner-detector)
37. [Detect Corners with Shi-Tomasi Corner Detector](#detect-corners-with-shi-tomasi-corner-detector)
38. [Background Subtraction Methods](#background-subtraction-methods)
39. [MeanShift Object Tracking](#meanshift-object-tracking)
40. [Object Tracking CamShift Method](#object-tracking-camshift-method)

---

### Introduction to OpenCV

OpenCV (Open Source Computer Vision Library) is a powerful library for image and video processing. It provides thousands of functions for tasks like image loading, display, transformations, and computer vision algorithms. In Python, OpenCV is used via the `cv2` module. Before using OpenCV, ensure you install it (e.g., `pip install opencv-python`) and import `cv2`. The library represents images as NumPy arrays.

```python
import cv2
print("OpenCV version:", cv2.__version__)
```

### Read, Write, and Display Images

To load an image from disk, use `cv2.imread()`. To display it, use `cv2.imshow()`. Always use `cv2.waitKey()` to keep the window open, and `cv2.destroyAllWindows()` to close it. To save an image, use `cv2.imwrite()`.

```python
import cv2

# Load an image (change the path to an actual image file)
image = cv2.imread('path/to/image.jpg', cv2.IMREAD_COLOR)
if image is None:
    print("Error: Image not found.")
else:
    cv2.imshow('Loaded Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_image.png', image)
```

### Read, Write, and Display Video from Camera

To capture live video from a camera, create a `VideoCapture` object (e.g., `0` for default webcam). Read frames in a loop and display them. To save video, create a `VideoWriter` object. Always release the capture and writer and destroy windows when done.

```python
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
else:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
```

### Draw Geometric Shapes on Images

OpenCV provides functions to draw lines, rectangles, circles, ellipses, and polygons on images. Below example creates a blank image and draws various shapes.

```python
import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
cv2.line(img, (0,0), (511,511), (255,0,0), 5)
cv2.rectangle(img, (384,0), (510,128), (0,255,0), 3)
cv2.circle(img, (447,63), 63, (0,0,255), -1)
cv2.ellipse(img, (256,256), (100,50), 0, 0, 180, (255,255,0), 2)
pts = np.array([[100,300],[200,200],[300,300]], np.int32).reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255), 2, cv2.LINE_AA)
cv2.imshow('Shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Setting Camera Parameters in OpenCV

You can get or set camera properties (brightness, contrast, resolution, etc.) using `cap.get()` and `cap.set()`. Properties are identified by constants like `cv2.CAP_PROP_BRIGHTNESS`, `cv2.CAP_PROP_FRAME_WIDTH`, etc.

```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    print(f"Width: {width}, Height: {height}, Brightness: {brightness}")
    cap.release()
```

### Show Date and Time on Videos

Overlay current date and time on video frames using `datetime`. Example:

```python
import cv2, datetime

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)
    cv2.imshow('Video with Timestamp', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
```

### Handle Mouse Events in OpenCV

Respond to mouse clicks on images by setting a callback:

```python
import cv2

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Left click at ({x}, {y})')

img = cv2.imread('image.jpg')
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', on_mouse)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### More Mouse Event Examples

Drawing circles where the user clicks:

```python
import cv2
import numpy as np

img = np.zeros((512,512,3), dtype=np.uint8)
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 20, (255,0,0), -1)

cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', draw_circle)
while True:
    cv2.imshow('Canvas', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
```

### Splitting and Merging Channels, Resizing, ROI, and Image Arithmetic

Separate color channels, merge them, resize images, select ROI, and perform arithmetic:

```python
import cv2

img = cv2.imread('image.jpg')
b, g, r = cv2.split(img)            # split channels
merged = cv2.merge([b, g, r])      # merge back
resized = cv2.resize(img, (300,300))
roi = img[100:200, 100:200]         # region of interest
added = cv2.add(img, img)          # image addition
weight = cv2.addWeighted(img, 0.6, resized, 0.4, 0)
```

### Bitwise Operations (AND, OR, XOR, NOT)

Mask one image with another mask image:

```python
mask = cv2.imread('mask.png', 0)
res_and = cv2.bitwise_and(img, img, mask=mask)
res_or = cv2.bitwise_or(img, img, mask=mask)
res_xor = cv2.bitwise_xor(img, img, mask=mask)
res_not = cv2.bitwise_not(img)
```

### Binding Trackbars to OpenCV Windows

Use trackbars to interactively adjust parameters:

```python
import cv2
import numpy as np

def nothing(x): pass

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Controls')
cv2.createTrackbar('R','Controls',0,255,nothing)
cv2.createTrackbar('G','Controls',0,255,nothing)
cv2.createTrackbar('B','Controls',0,255,nothing)
while True:
    r = cv2.getTrackbarPos('R','Controls')
    g = cv2.getTrackbarPos('G','Controls')
    b = cv2.getTrackbarPos('B','Controls')
    img[:] = [b, g, r]
    cv2.imshow('Controls', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
```

### Object Detection and Tracking Using HSV Color Space

Detect an object of specific color:

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (50, 100, 100)
    upper = (70, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
```

### Simple Image Thresholding

Apply a fixed-level threshold:

```python
import cv2

img = cv2.imread('gray.jpg', 0)
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Adaptive Thresholding

Use local neighborhood thresholds:

```python
import cv2

img = cv2.imread('gray.jpg', 0)
adapt = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY,11,2)
cv2.imshow('Adaptive', adapt)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Matplotlib with OpenCV

Display images in plots:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('RGB Image')
plt.axis('off')
plt.show()
```

### Morphological Transformations

Erode, dilate, open, and close:

```python
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
eroded = cv2.erode(mask, kernel, iterations=1)
dilated = cv2.dilate(mask, kernel, iterations=1)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### Smoothing/Blurring Images

Apply averaging, Gaussian, median, and bilateral filters:

```python
import cv2

blur = cv2.blur(img,(5,5))
gauss = cv2.GaussianBlur(img,(5,5),0)
median = cv2.medianBlur(img,5)
bilateral = cv2.bilateralFilter(img,9,75,75)
```

### Image Gradients and Edge Detection

Use Sobel, Scharr, and Laplacian operators:

```python
import cv2

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
lap = cv2.Laplacian(gray,cv2.CV_64F)
```

### Canny Edge Detection

Detect edges using Canny:

```python
import cv2

edges = cv2.Canny(gray,100,200)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Image Pyramids

Build Gaussian and Laplacian pyramids:

```python
import cv2

lower_reso = cv2.pyrDown(img)
higher_reso = cv2.pyrUp(lower_reso)
```

### Image Blending using Pyramids

Blend two images with masks and pyramids:

```python
# build pyramids for img1, img2, and mask, then combine levels
```

### Find and Draw Contours

Find object contours:

```python
import cv2

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
```

### Motion Detection and Tracking Using Contours

Detect movement between frames:

```python
# use frame differencing, threshold, findContours
```

### Detect Simple Geometric Shapes

Approximate contours to polygons and classify shapes:

```python
# for each contour: approx=cv2.approxPolyDP, then len(approx)
```

### Understanding Image Histograms

Compute and plot histograms:

```python
import cv2
import matplotlib.pyplot as plt

hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()
```

### Template Matching using OpenCV

Find template in image:

```python
import cv2

res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
```

### Hough Line Transform Theory

Detect straight lines in images using the Hough transform algorithm.

### Hough Line Transform using HoughLines

```python
import cv2

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[:,0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0,y0 = a*rho, b*rho
    pt1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
    pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
    cv2.line(img,pt1,pt2,(0,0,255),2)
```

### Probabilistic Hough Line Transform using HoughLinesP

```python
linesP = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)
for x1,y1,x2,y2 in linesP[:,0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
```

### Road Lane Line Detection (Part 1)

Detect edges, define ROI, mask, and find lane lines.

### Road Lane Line Detection (Part 2)

Separate left/right lines, fit with `np.polyfit`, and draw.

### Road Lane Line Detection (Part 3)

Merge lanes back and overlay on original image.

### Circle Detection using Hough Circle Transform

```python
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,20,
                           param1=50,param2=30,minRadius=0,maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x,y,r in circles[0,:]:
        cv2.circle(img,(x,y),r,(0,255,0),2)
```

### Face Detection using Haar Cascade Classifiers

```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
```

### Eye Detection using Haar Cascade Classifiers

```python
eyecascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes = eyecascade.detectMultiScale(gray)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
```

### Detect Corners with Harris Corner Detector

```python
corners = cv2.cornerHarris(gray,2,3,0.04)
img[corners>0.01*corners.max()]=[0,0,255]
```

### Detect Corners with Shi-Tomasi Corner Detector

```python
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
for c in corners:
    x,y = c.ravel()
    cv2.circle(img,(x,y),5,(0,0,255),-1)
```

### Background Subtraction Methods

Use MOG2 and KNN:

```python
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(frame)
```

### MeanShift Object Tracking

```python
# initialize track window and ROI histogram, then cv2.meanShift
```

### Object Tracking CamShift Method

```python
# same as MeanShift but use cv2.CamShift for rotated windows
```

---

*End of OpenCV Python Tutorial*
