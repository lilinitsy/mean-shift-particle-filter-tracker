class BoundingCircle:
	center_x: int
	center_y: int
	radius: int

	def __init__(self, x: int, y: int, r: int, max_x: int, max_y: int, min_x: int, min_y: int):
		self.center_x = x
		self.center_y = y

		if self.center_x - r < min_x or self.center_y - r < min_y:
			if self.center_x < self.center_y:
				self.radius = self.center_x - 1
			else:
				self.radius = self.center_y - 1
		
		elif self.center_x + r >= max_x or self.center_y + r >= max_y:
			if self.center_x < self.center_y:
				self.radius = self.center_x - 1
			else:
				self.radius = self.center_y - 1
		
		else:
			self.radius = r
