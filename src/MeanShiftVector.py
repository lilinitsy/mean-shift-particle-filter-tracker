#!/usr/bin/python
''' Constructs a videocapture device on either webcam or a disk movie file.
Press q to exit

Junaed Sattar
October 2018
'''
from __future__ import division
import numpy as np
import cv2
import sys
import random
import math

from matplotlib import pyplot as plt
from typing import List

from BoundingBox import BoundingBox
from Particles import Particles

'''global data common to all vision algorithms'''
'''Mr. Global Arrays would be proud'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageWidth = 0
imageHeight = 0

mouse_x = 0
mouse_y = 0

width = 60
height = 60
current_x = 0
current_y = 0

hacky_click_has_occurred = False


'''
	Defines a color model for the target of interest.
	rn: Now, just reading pixel color at location
'''

def TuneTracker(x, y):
	global r, g, b, image
	b, g, r = image[y, x]
	print(r, g, b, 'at location ', x, y)


''' Have to update this to perform Sequential Monte Carlo
	tracking, i.e. the particle filter steps.

	Currently this is doing naive color thresholding.
'''
def doTracking() -> None:
	global isTracking, image, r, g, b
	if isTracking:
		print(image.shape)
		imheight, imwidth, implanes = image.shape
		for j in range(imwidth):
			for i in range(imheight):
				bb, gg, rr = image[i, j]
				sumpixels = float(bb) + float(gg) + float(rr)
				if sumpixels == 0:
					sumpixels = 1
				if rr / sumpixels >= r and gg / sumpixels >= g and bb / sumpixels >= b:
					image[i, j] = [255, 255, 255]
				else:
					image[i, j] = [0, 0, 0]			   
	

# Mouse Callback function
def clickHandler(event, x, y, flags, param) -> None:
	global mouse_x
	global mouse_y
	global current_x
	global current_y
	global hacky_click_has_occurred

	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		mouse_x = x
		mouse_y = y
		current_x = x
		current_y = y
		hacky_click_has_occurred = True
		TuneTracker(x, y)


def mapClicks(x, y, curWidth, curHeight) -> None:
	global imageHeight, imageWidth
	imageX = x * imageWidth / curWidth
	imageY = y * imageHeight / curHeight
	return (imageX, imageY)


def draw_bounding_box(bounding_box: BoundingBox, image, color: (int, int, int)) -> None:
	x = bounding_box.bottomleft_x
	y = bounding_box.bottomleft_y
	w = bounding_box.width
	h = bounding_box.height
	x = int(x)
	y = int(y)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


def calculate_histogram(bounding_box: BoundingBox, img):
	min_x = bounding_box.bottomleft_x
	max_x = bounding_box.bottomleft_x + bounding_box.width
	min_y = bounding_box.bottomleft_y
	max_y = bounding_box.bottomleft_y + bounding_box.height
	
	# make a mask and get histogram in this window
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[int(min_x):int(max_x), int(min_y):int(max_y)] = 255
	histogram = cv2.calcHist([img], [0], mask, [256], [0, 256])
	return histogram

def draw(window, image):
	(bottom_x, bottom_y, width, height) = window
	#bounds = cv2.rectangle(image, (bottom_x, bottom_y), (bottom_x + width, bottom_y + height), 255, 2)
	#cv2.imshow('bounding_box', bounds)
			##img2 = cv2.rectangle(image, (x,y), (x+w,y+h), 255,2)
	cv2.rectangle(image, (bottom_x, bottom_y), (bottom_x + width, bottom_y + height), (0, 0, 255), 2)
	


def captureVideo(src) -> None:
	global image, isTracking, trackedImage, current_x, current_y

	cap = cv2.VideoCapture(src)
	if cap.isOpened() and src=='0':
		ret = cap.set(3, 640) and cap.set(4, 480)
		if ret == False:
			print('Cannot set frame properties, returning')
			return
	else:
		frate = cap.get(cv2.CAP_PROP_FPS)
		print(frate, ' is the framerate')
		waitTime = int(1000 / frate)

#	waitTime = time/image. Adjust accordingly.
	if src == 0:
		waitTime = 1
	if cap:
		print('Succesfully set up capture device')
	else:
		print('Failed to setup capture device')

	windowName = 'Input View, press q to quit'
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, clickHandler)

	ret,image = cap.read()
	#track_window = (mouse_x, mouse_y, width, height)
	termination_parameters = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 5)

	while(True):
		ret, image = cap.read()
		
		if(hacky_click_has_occurred):
			track_window = (current_x, current_y, width, height)
			tracking_region = image[mouse_y:mouse_y + height, mouse_x:mouse_x + width]
			mask = cv2.inRange(tracking_region, np.array((0.0, 0.0, 0.0)), np.array((255.0, 255.0, 255.0)))
			tracking_region_hist = cv2.calcHist([tracking_region], [0], mask, [256], [0,256])
			cv2.normalize(tracking_region_hist, tracking_region_hist, 0, 255, cv2.NORM_MINMAX)

			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([hsv], [0], tracking_region_hist, [0,256], 1)
			# apply meanshift to get the new location
			ret, track_window = cv2.meanShift(dst, track_window, termination_parameters)
			current_x = track_window[0]
			current_y = track_window[1]
			draw(track_window, image)
		
		# Display the resulting frame   
		if isTracking:
			doTracking()
		cv2.imshow(windowName, image )										
		inputKey = cv2.waitKey(waitTime) & 0xFF
		if inputKey == ord('q'):
			break
		elif inputKey == ord('t'):
			isTracking = not isTracking			
		elif inputKey == ord('h'):
			#plt.plot(target_histogram)
			''''''

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


print('Starting program')
if __name__ == '__main__':
	arglist = sys.argv
	src = 0
	print('Argument count is ', len(arglist))
	if len(arglist) == 2:
		src = arglist[1]
	else:
		src = 0
	captureVideo(src)
else:
	print('Not in main')
	
