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


from BoundingBox import BoundingBox

'''global data common to all vision algorithms'''
'''Mr. Global Arrays would be proud'''
isTracking = False
r = g = b = 0.0
image = np.zeros((640, 480, 3), np.uint8)
trackedImage = np.zeros((640, 480, 3), np.uint8)
imageWidth = imageHeight = 0

mouse_x = mouse_y = 0

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
def doTracking():
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
def clickHandler(event, x, y, flags, param):
	global mouse_x
	global mouse_y
	if event == cv2.EVENT_LBUTTONUP:
		print('left button released')
		mouse_x = x
		mouse_y = y
		TuneTracker(x, y)


def mapClicks(x, y, curWidth, curHeight):
	global imageHeight, imageWidth
	imageX = x * imageWidth / curWidth
	imageY = y * imageHeight / curHeight
	return imageX, imageY


def propagate_step(M, bounding_box):
	windows = []
	for i in range(0, M):
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

def captureVideo(src):
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
	
	
	while(True):
		# Capture frame-by-frame
		ret, image = cap.read()
		if ret == False:
			break
		
		
		bounding_box = BoundingBox(mouse_x, mouse_y, 30, 20, 640, 480, 0, 0)
		bounding_box.print()
		propagate_step(20, bounding_box)
		# Display the resulting frame   
		if isTracking:
			doTracking()
		cv2.imshow(windowName, image )										
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
	
