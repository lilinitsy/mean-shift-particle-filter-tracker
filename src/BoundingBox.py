import numpy as np


class BoundingBox:
#	(center_x, center_y) = (0, 0)
#	width = 0
#	height = 0
	
	def __init__(self, x, y, w, h, max_x, max_y):
		(self.center_x, self.center_y) = (x, y)

		if(x + w >= max_x):
			self.width = max_x - x - 1
		elif(x - w < 0):
			self.width = x + 1
		else:
			self.width = w

		if(y + h >= max_y):
			self.height = max_y - y - 1
		elif(y - h < 0):
			self.height = y + 1
		else:
			self.height = h
	
	def print(self):
		print("center: (", self.center_x, self.center_y, ")")
		print("width: ", self.width, self.height)
		