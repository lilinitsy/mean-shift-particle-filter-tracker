from __future__ import division
import numpy as np
import cv2
import sys
import random

from matplotlib import pyplot as plt
from typing import List


img = cv2.imread('../test_images/jt.jpg', 0)

mask = np.zeros(img.shape[:2], np.uint8)
mask[121:125, 223:227] = 255

hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
plt.plot(hist_mask)

plt.show()