from BoundingBox import BoundingBox

class Particles:
	def __init__(self, bbox: BoundingBox, weight: float):
		self.bounding_box = bbox
		self.weight = weight