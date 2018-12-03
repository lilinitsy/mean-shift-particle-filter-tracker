import numpy as np

from BoundingBox import BoundingBox

class Particles:
	def __init__(self, bbox: BoundingBox, hist: np.ndarray, weight: float):
		self.bounding_box = bbox
		self.weight = weight
		self.histogram = hist