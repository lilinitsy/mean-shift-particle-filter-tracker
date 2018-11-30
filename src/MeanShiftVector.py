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
imageWidth = imageHeight = 0

mouse_x = mouse_y = 0

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
	global hacky_click_has_occurred

	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		mouse_x = x
		mouse_y = y
		hacky_click_has_occurred = True
		TuneTracker(x, y)


def mapClicks(x, y, curWidth, curHeight) -> None:
	global imageHeight, imageWidth
	imageX = x * imageWidth / curWidth
	imageY = y * imageHeight / curHeight
	return imageX, imageY

def create_particles(bounding_box: BoundingBox, num_windows: int) -> List[Particles]:
	particles = []
	for i in range(0, num_windows):
		center_x = random.randint(
			bounding_box.center_x - bounding_box.width,
			bounding_box.center_x + bounding_box.width)
		center_y = random.randint(
			bounding_box.center_y - bounding_box.height,
			bounding_box.center_y + bounding_box.height)
		width = bounding_box.width / 10
		height = bounding_box.height / 10
		max_x = center_x + width
		max_y = center_y + height
		min_x = center_x - width
		min_y = center_y - height

		bbox = BoundingBox(center_x, center_y, width, height, max_x, max_y, min_x, min_y)
		weight = 1 # uniform weight
		particle = Particles(bbox, weight)
		particles.append(particle)

	return particles



def sliding_window_histograms(bounding_box: BoundingBox, particles: Particles) -> (np.ndarray, List):
	histograms = [[0 for i in range(0, 3)] for j in range(0, len(particles))]
	bounding_boxes = []

	for i in range(0, len(particles)):
		mask = np.zeros(image.shape[:2], np.uint8)
		min_x = particles[i].bounding_box.center_x - particles[i].bounding_box.width
		max_x = particles[i].bounding_box.center_x + particles[i].bounding_box.width
		min_y = particles[i].bounding_box.center_y - particles[i].bounding_box.height
		max_y = particles[i].bounding_box.center_y + particles[i].bounding_box.height
		bbox = BoundingBox(particles[i].bounding_box.center_x,
			particles[i].bounding_box.center_y,
			particles[i].bounding_box.width,
			particles[i].bounding_box.height,
			640, 480, 0, 0)
		bounding_boxes.append(bbox)


		mask[int(min_x):int(max_x), int(min_y):int(max_y)] = 255
		
		#histograms[i].append(histogram)
		colours = ('b', 'g', 'r')
		for j, cols in enumerate(colours):
			histograms[i][j] = cv2.calcHist([image], [j], mask, [16], [0, 256])
	return (histograms, bounding_boxes)

# shape: rows, columns, channels (rgb)
# 3-2: Compute colour histogram for each particle
def generate_histograms(bounding_box: BoundingBox) -> np.ndarray:
	histograms = []	
	min_x = bounding_box.center_x - bounding_box.width / 2
	max_x = bounding_box.center_x + bounding_box.width / 2
	min_y = bounding_box.center_y - bounding_box.height / 2
	max_y = bounding_box.center_y + bounding_box.height / 2
	print("minx: ", min_x)
	print("maxx: ", max_x)
	print("miny: ", min_y)
	print("maxy: ", max_y)
	print("\n")
	
	# make a mask and get histogram in this window
	mask = np.zeros(image.shape[:2], np.uint8)
	mask[int(min_x):int(max_x), int(min_y):int(max_y)] = 255

	colours = ('b', 'g', 'r')
	for i, cols in enumerate(colours):
		histogram = cv2.calcHist([image], [i], mask, [256], [0, 256])
		histograms.append(histogram)

	# numpy.ndarray	
	return histograms


# Not going to type annotate this yet
def create_kernel(bounding_box):
	x = bounding_box.width
	y = bounding_box.height
	kernel = [[0 for i in range(y)] for j in range(x)]

	for i in range(int(x / 2)):
		xdist = x / 2 - i
		for j in range(int(y / 2)):
			ydist = y / 2 - j
			distance = math.sqrt(xdist ** 2 + ydist ** 2)
			if(distance > 0):
				kernel[i][j] = 1 / distance
			else:
				kernel[i][j] = 1
	
	for i in range(int(x / 2), x):
		xdist = x - i
		for j in range(int(y / 2), y):
			ydist = y / 2 - j
			distance = math.sqrt(xdist ** 2 + ydist ** 2)
			if(distance > 0):
				kernel[i][j] = 1 / distance
			else:
				kernel[i][j] = 1

	return kernel

# 3-3: Similiarity step, use Bhattacharyya Distance for PF
# But for MSV, use the colour histogram similarity
#def tracking_histogram_routine(target_histogram, bounding_box, kernel):



def draw_bounding_box(bounding_box: BoundingBox, image, color: (int, int, int)) -> None:
	x = bounding_box.center_x - bounding_box.width / 2
	y = bounding_box.center_y - bounding_box.height / 2
	w = bounding_box.width
	h = bounding_box.height
	x = int(x)
	y = int(y)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)



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
	#tracking_histogram = target_histogram
	window_histograms = []
	window_bbs = []

	while(True):
		# Capture frame-by-frame
		ret, image = cap.read()
		if ret == False:
			break
		
		if(hacky_click_has_occurred):
			bounding_box = BoundingBox(mouse_x, mouse_y, 30, 30, 640, 480, 0, 0)
			kernel = create_kernel(bounding_box)
			particles = create_particles(bounding_box, 20)
			
			(window_histograms, window_bbs) = sliding_window_histograms(bounding_box, particles)
			target_histograms = generate_histograms(bounding_box)
			#tracking_histogram = tracking_histogram_routine(target_histogram, bounding_box, kernel)



			draw_bounding_box(bounding_box, image, color = (0, 0, 255))
			#draw_bounding_box(track_hist_box, src, color = "green")


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
	
