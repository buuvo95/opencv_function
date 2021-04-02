import cv2 as cv
import numpy as np

# Translation: Shift image x pixels and y pixels
# -x --> Left
# -y --> Up
# x --> Right
# y --> Down
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# Rotation Image by defined angle
def rotate(img, angle, rotPoint=None):
    (height, width)= img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0) # 1.0 is scaled value
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

# Contour Detection
# Basiclly, Contour is a boundary of object: 
# Contour is usefull with Object detection and recognition
def countourDetection(img):
    # Convert image to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Canny Detector - Canny Edge
    canny = cv.Canny(gray, 125, 175)

    # We can detect the contours by thread
    # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    # contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Find Contour
    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Draw Contour image
    # Define blank image
    blank = np.zero(img.shape[:2], dtype='unit8')
    cv2.drawContours(blank, contours, -1, (0,0,255), 2) # -1: all contour, (0,0,255): corlor, 2: thickness

    return contours

# Split image into blue - green -red
def split(img):
    b,g,r = cv.split(img)
    return b, g, r

# Merge blue - green - red onto an image
def merge(b,g,r):
    img = cv2.merge([b,g,r])
    return img

# Blurring image
def blurring(img):
    # Averaing Blur
    average = cv.blur(img, (7,7))

    #Gaussian Blur
    gauss = cv.GaussianBlur(img, (7,7), 0)

    # Median blur
    median = cv.medianBlur(img, 3)

    # Bilateral
    bilateral = cv.bilateralFilter(img, 5, 15, 15)

# Advanced Method
# BITWISE OPERATIONs
def bitwise():
    blank = np.zeros((400,400), dtype='unit8'))

    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
    circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

    # Bitwise AND
    bitwise_and = cv.bitwise_and(rectangle, circle)
    # Bitwise OR
    bitwise_or = cv.bitwise_or(rectangle, circle)
    # Bitwire XOR
    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    # Bitwise NOT
    bitwise_not = cv.bitwise_not(rectangle)

# Masking
def masking(img):
    blank = np.zeros(img.shape[:2], dtype='uint8')

    mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

    masked = cv.bitwise_and(img, img, mask=mask)

# Histogram computing
def histogram():
    
