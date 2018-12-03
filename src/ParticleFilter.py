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


'''
	REDEFINE BOUNDING BOX TO BE BOUNDING CIRCLE?
	MAKE THEM CENTERED AT RANDOM LOCATIONS THROUGHOUGHT THE IMAGE
	GET THE AVERAGE POSITION (and show it in a different colour)
'''


'''global data common to all vision algorithms'''
'''Mr. Global Arrays would be proud'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageWidth = imageHeight = 0

mouse_x = mouse_y = 0

hacky_click_has_occurred = False
	   
	

# Mouse Callback function
def clickHandler(event, x, y, flags, param) -> None:
	global mouse_x
	global mouse_y
	global hacky_click_has_occurred

	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		mouse_x = x
		mouse_y = y
		hacky_click_has_occurred = True


def mapClicks(x, y, curWidth, curHeight) -> None:
	global imageHeight, imageWidth
	imageX = x * imageWidth / curWidth
	imageY = y * imageHeight / curHeight
	return imageX, imageY

def create_particles(boun: BoundingBox, num_windows: int) -> List[Particles]:
	particles = []

	return particles



def sliding_window_histograms(bounding_box: BoundingBox, particles: Particles) -> (np.ndarray, List):
	histograms = [[0 for i in range(0, 3)] for j in range(0, len(particles))]
	bounding_boxes = []


	return (histograms, bounding_boxes)

# shape: rows, columns, channels (rgb)
# 3-2: Compute colour histogram for each particle
def generate_histograms(bounding_box: BoundingBox) -> np.ndarray:
	histograms = []	

	# numpy.ndarray	
	return histograms


# Not going to type annotate this yet
def create_kernel(bounding_box):
	x = int(bounding_box.width)
	y = int(bounding_box.height)
	kernel = [[0 for i in range(y)] for j in range(x)]
	# use gaussian distance???
	return kernel

# 3-3: Similiarity step, use Bhattacharyya Distance for PF
# But for MSV, use the colour histogram similarity


def draw_bounding_box(bounding_box: BoundingBox, image, color: (int, int, int)) -> None:
	x = bounding_box.bottomleft_x
	y = bounding_box.bottomleft_y
	w = bounding_box.width
	h = bounding_box.height
	x = int(x)
	y = int(y)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


def tracking_histogram_routine(target_histograms, bounding_box: BoundingBox, kernel, window_histograms: List, window_bbs: List, window_kernels: List, current_center: (int, int)):
	# Similarity measure for each window_histogram against target_histogram
	# Find the most similar histogram
	# Set that as the center, check if distance from current_center to the best window_histogram is below a distance
	# If it isn't, make a new bounding box using window_histogram as the center,
	# Recurse with target_histogram, new_bounding_box, kernel, new_window_histograms, new_window_bbs, window_kernels, new_center
	max = 0
	index = 0
	print("max: ", max)
	print("index: ", index)
		


def captureVideo(src) -> None:
	global image, isTracking, trackedImage

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

#	waitTime = time/frame. Adjust accordingly.
	if src == 0:
		waitTime = 1
	if cap:
		print('Succesfully set up capture device')
	else:
		print('Failed to setup capture device')

	windowName = 'Input View, press q to quit'
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, clickHandler)
	
	bounding_box = BoundingBox(mouse_x, mouse_y, 30, 30, 640, 480, 0, 0)
	kernel = create_kernel(bounding_box)


	target_histograms = generate_histograms(bounding_box)
	#tracking_histogram = np.zeros()
	window_histograms = []
	window_bbs = []

	while(True):
		# Capture frame-by-frame
		ret, image = cap.read()
		if ret == False:
			break

		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		if(hacky_click_has_occurred):
			bounding_box = BoundingBox(mouse_x, mouse_y, 30, 30, 640, 480, 0, 0)
			kernel = create_kernel(bounding_box)
			particles = create_particles(bounding_box, 20)
			
			(window_histograms, window_bbs) = sliding_window_histograms(bounding_box, particles)
			
			window_kernels = []
			for i in range(0, len(window_bbs)):
				window_kernel = create_kernel(window_bbs[i])
				window_kernels.append(window_kernel)


			target_histograms = generate_histograms(bounding_box)
			#tracking_histogram = tracking_histogram_routine(target_histogram, bounding_box, kernel)
			tracking_histogram_routine(target_histograms, bounding_box, kernel, window_histograms, window_bbs, window_kernels, (bounding_box.bottomleft_x, bounding_box.bottomleft_y))


			draw_bounding_box(bounding_box, image, color = (0, 0, 255))
			#draw_bounding_box(track_hist_box, image, color = (0, 255, 0))


		# Display the resulting frame   
		cv2.imshow(windowName, image )										
		inputKey = cv2.waitKey(waitTime) & 0xFF
		if inputKey == ord('q'):
			break
		elif inputKey == ord('t'):
			isTracking = not isTracking			
		elif inputKey == ord('h'):
			#plt.plot(target_histogram)
			colours = ('b', 'g', 'r')
			for i, col in enumerate(colours):
				#plt.plot(target_histograms[i], color = col)
				plt.plot(window_histograms[0][i], color = col)
				plt.hist(window_histograms[0][1], color = col)
				


		plt.show()
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
	
