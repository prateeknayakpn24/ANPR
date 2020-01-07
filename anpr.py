import pytesseract
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
def show(image):
    cv2.imshow('image',image)
    cv2.waitKey(5000)

def auto_canny(image,sigma=0.10):
    # compute the median of the single channel pixel intensities
    v=np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower=int(max(0,(1.0-sigma)*v))
    upper=int(min(255,(1.0+sigma)*v))
    edged=cv2.Canny(image,lower,upper)
    # return the edged image
    return edged


frame=cv2.imread(".\mercedes.jpg")
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

edge = auto_canny(blur)

conts,new = cv2.findContours(edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
frame_copy=frame.copy()
_=cv2.drawContours(frame_copy,conts,-1,(255,0,255),2)
show(frame_copy)