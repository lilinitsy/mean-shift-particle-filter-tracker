import numpy as np


class BoundingBox:
	bottomleft_x: int
	bottomleft_y: int
	width: int
	height: int

	def __init__(self, x: int, y: int, w: int, h: int, max_x: int, max_y: int, min_x: int, min_y: int):
		(self.bottomleft_x, self.bottomleft_y) = (x, y)

		if(x + w >= max_x):
			self.width = max_x - x - 1
		elif(x - w < min_x):
			self.width = x + 1
		else:
			self.width = w

		if(y + h >= max_y):
			self.height = max_y - y - 1
		elif(y - h < min_y):
			self.height = y + 1
		else:
			self.height = h
	
	def print(self):
		print("center: (", self.bottomleft_x, self.bottomleft_y, ")")
		print("width: ", self.width, self.height)
		