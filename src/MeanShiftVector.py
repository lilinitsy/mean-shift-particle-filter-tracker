#!/usr/bin/python
''' Constructs a videocapture device on either webcam or a disk movie file.
Press q to exit

Original boiler plate code (mouse events, window capture) by Junaed Sattar
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

width = 60
height = 60
current_x = []
current_y = []
hacky_click_has_occurred = False
	

# Mouse Callback function
def clickHandler(event, x, y, flags, param) -> None:
	global current_x
	global current_y
	global hacky_click_has_occurred

	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		current_x.append(x)
		current_y.append(y)
		hacky_click_has_occurred = True


def calculate_histogram(bounding_box: BoundingBox, img):
	min_x = bounding_box.bottomleft_x
	max_x = bounding_box.bottomleft_x + bounding_box.width
	min_y = bounding_box.bottomleft_y
	max_y = bounding_box.bottomleft_y + bounding_box.height
	
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[int(min_x):int(max_x), int(min_y):int(max_y)] = 255
	histogram = cv2.calcHist([img], [0], mask, [256], [0, 256])
	return histogram


def draw(window, image, colour: (int, int, int)) -> None:
	(bottom_x, bottom_y, width, height) = window
	cv2.rectangle(image, (bottom_x, bottom_y), (bottom_x + width, bottom_y + height), colour, 2)
	

def captureVideo(src) -> None:
	global image, isTracking, trackedImage, current_x, current_y

	cap = cv2.VideoCapture(src)
	if cap.isOpened() and src == '0':
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

	windowName = 'Mean Shift Vector, press q to quit'
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, clickHandler)

	ret, image = cap.read()
	
	# OpenCV docs - first 5 is the # iterations, second 5 is the min pixels to move before stopping... lower means more accurate?
	termination_parameters = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 5)

	while(True):
		ret, image = cap.read()
		
		if(hacky_click_has_occurred):
			for i in range(0, len(current_x)):
				track_window = (current_x[i], current_y[i], width, height)
				tracking_region = image[current_y[i]:current_y[i] + height, current_x[i]:current_x[i] + width]
				mask = cv2.inRange(tracking_region, np.array((0.0, 0.0, 0.0)), np.array((180.0, 180.0, 180.0)))
				tracking_region_hist = cv2.calcHist([tracking_region], [0], mask, [180], [0,180])
				cv2.normalize(tracking_region_hist, tracking_region_hist, 0, 180, cv2.NORM_MINMAX)

				hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
				back_propagation = cv2.calcBackProject([hsv_image], [0], tracking_region_hist, [0,180], 1)
				ret, track_window = cv2.meanShift(back_propagation, track_window, termination_parameters)
				current_x[i] = track_window[0]
				current_y[i] = track_window[1]
				draw(track_window, image, (0, 255, 0))
		
		# Display the resulting frame   
		cv2.imshow(windowName, image)										

		inputKey = cv2.waitKey(waitTime) & 0xFF
		if inputKey == ord('q'):
			break
		elif inputKey == ord('t'):
			isTracking = not isTracking			

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
	
