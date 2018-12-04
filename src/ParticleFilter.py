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
current_x = 0
current_y = 0

hacky_click_has_occurred = False
particles = []
	   
	
# Mouse Callback function
def clickHandler(event, x, y, flags, param) -> None:
	global mouse_x
	global mouse_y
	global hacky_click_has_occurred
	global particles

	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		mouse_x = x
		mouse_y = y
		current_x = mouse_x
		current_y = mouse_y
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)	
		particles = create_particles(20, 640, 480, 20, 20, hsv_image)
		hacky_click_has_occurred = True


def create_particles(num_particles: int, max_x: int, max_y: int, width: int, height: int, image) -> List[Particles]:
	particles = []

	for i in range(0, num_particles):
		x = random.randint(width, max_x - 1 - width)
		y = random.randint(height, max_y - 1 - height)
		bbox = BoundingBox(x, y, width, height, 640, 480, 0, 0)
		
		histogram = generate_histograms(bbox, image)
		particle = Particles(bbox, histogram, 1)
		particles.append(particle)

	return particles


# shape: rows, columns, channels (rgb)
# 3-2: Compute colour histogram for each particle
def generate_histograms(bounding_box: BoundingBox, image) -> np.ndarray:
	min_x = bounding_box.bottomleft_x
	min_y = bounding_box.bottomleft_y
	max_x = bounding_box.bottomleft_x + bounding_box.width
	max_y = bounding_box.bottomleft_y + bounding_box.height

	mask = np.zeros(image.shape[:2], np.uint8)
	mask[int(min_x):int(max_x), int(min_y):int(max_y)] = 255
	histogram = cv2.calcHist([image], [0], mask, [180], [0, 180])
	return histogram


def similarity(particle_histogram, target_histogram) -> float:
	similarity_score = cv2.compareHist(particle_histogram, target_histogram, cv2.HISTCMP_INTERSECT)
	return similarity_score


def draw_bounding_box(bounding_box: BoundingBox, image, colour: (int, int, int)) -> None:
	x = bounding_box.bottomleft_x
	y = bounding_box.bottomleft_y
	width = bounding_box.width
	height = bounding_box.height
	cv2.rectangle(image, (x, y), (x + width, y + height), colour, 2)


def captureVideo(src) -> None:
	global image, isTracking, trackedImage, particles, current_x, current_y

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

	while(True):
		# Capture frame-by-frame
		ret, image = cap.read()
		if ret == False:
			break

		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		if(hacky_click_has_occurred):
			# Particles is a global, modified in def mouseClicks
			target_bounding_box = BoundingBox(current_x, current_y, 20, 20, 640, 480, 0, 0)
			target_histogram = generate_histograms(target_bounding_box, hsv_image)

			# have to normalize before comparisons
			cv2.normalize(target_histogram, target_histogram, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
			for i in range(0, len(particles)):
				cv2.normalize(particles[i].histogram, particles[i].histogram, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)

			particle_weights = 0
			(position_x, position_y) = (0, 0)
			for i in range(0, len(particles)):
				similarity_score = similarity(particles[i].histogram, target_histogram)
				particle_weights += similarity_score
				(center_x, center_y) = particles[i].bounding_box.get_center()
				position_x += center_x * similarity_score
				position_y += center_y * similarity_score
				print("i: ", i, "\tSimilarity Score: ", similarity_score)
			
			position_x /= particle_weights
			position_y /= particle_weights
			position_x = int(position_x)
			position_y = int(position_y)
			current_x = position_x
			current_y = position_y
			tracking_bounding_box = BoundingBox(position_x, position_y, 20, 20, 640, 480, 0, 0)
			draw_bounding_box(tracking_bounding_box, image, (255, 0, 0))

			for i in range(0, len(particles)):
				draw_bounding_box(particles[i].bounding_box, image, (0, 255, 0))
			draw_bounding_box(target_bounding_box, image, (255, 255, 0))
		# Display the resulting frame   
		cv2.imshow(windowName, image)					
		inputKey = cv2.waitKey(waitTime) & 0xFF
		if inputKey == ord('q'):
			break
		elif inputKey == ord('t'):
			isTracking = not isTracking							


		plt.show()
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


print('Starting program')
if __name__ == '__main__':
	random.seed()
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
	
