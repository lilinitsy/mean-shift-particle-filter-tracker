# mean-shift-particle-filter-tracker

## Dependencies
- OpenCV
- Python3

## Running
- To run the particle filter, run ``python ParticleFilter.py``
- To run the mean shift vector, run ``python MeanShiftVector.py``
- (For each of these, the command may begin with ``python3`` if that is the default python command invokes Python 2)
- Click on the target region that you wish to track.
- Videos exist in the /videos folder

## Real World Considerations
Hue was used instead of BGR values because comparing three colour histograms would be both more tedious to program, and could, when tracking a large number of objects, slow down the performance. 
Hue only does perform worse.

Issues common to both the MSV and PF trackers are that when using hue values, there is only one criterion; this can yield more error.
In the videos, it can be seen that the trackers seem to like my dark blue shirt as a substitute for my dark gray phone case; I believe this is because the hue values are similar.
180 bins (Hue is 0-180) are also used. 
I chalk this up to having more criterion for a match.

Both run at around 27-31 fps, and adding 14+ trackers to the mean shift vector did not affect the performance. The framerate seems bound by the webcam capture - closing my screen around halfway lowers the fps to 14-16, but that is still independent of the trackers.



## Mean Shift Vector
The mean shift tracker works by defining the window as a 30x30 bounding box surrounding where the user clicked.
With each iteration, the Hue histogram is calculated, and this histogram is compared to the original histogram that was made when the user clicked. 
The part of the histogram of the "track_window" variable that is closest to the original histogram is then set as the best location, and the mean shift vector switches over.
The bounding box is green.

The mean shift tracker is implemented to track n objects.
This was easy to implement, by just making everything that was an int or a histogram into a list of ints or histograms.
Oddly, reacquiring tracking boxes after occlusion or losing them is harder when there are 2 or more tracked objects; this makes little sense to me, but the accuracy with 1 tracker is much better than the particle filter's.

The mean shift vector performed better when using 9 bins for hue instead of 180; this is unlike the Particle Filter, which makes no sense to me. The 9 bin recording can be seen in "videos/mean_shift_normal.flv"


## Particle Filter
The particle filter tracker isn't implemented exactly like the drone tracker shown. 
The drone tracker presumably samples from the entire world originally, and then converges the particles onto the target location. 
This implementation samples as a sliding window within a bounding box from the user click. 
It will then resample the pixels within this bounding box (translated to be centered at their average position), using the sliding window method.

A similarity metric is assigned based on the similarity of the histograms using an OpenCV function. Then, the average position factoring in the weights is determined.
The weighting is done in this block of code:
```
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
```
This code has the chance to rarely give a division by zero error, if the similarity scores for all histograms are 0. It is fixed with an if/else check.
The accuracy is not especially bad, but if the tracker loses the target, it seems to drift to a very specific part of the window - bottom, and around 3/4ths of the way to the right.
