import cv2
import numpy as np
import math
import copy
from collections import defaultdict

def segment_by_angle_kmeans(lines, k=2, **kwargs):
	"""Groups lines based on angle with k-means.

	Uses k-means on the coordinates of the angle on the unit circle 
	to segment `k` angles inside `lines`.
	"""

	# Define criteria = (type, max_iter, epsilon)
	default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
	criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
	flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
	attempts = kwargs.get('attempts', 10)

	# returns angles in [0, pi] in radians
	angles = np.array([line[0][1] for line in lines])
	# multiply the angles by two and find coordinates of that angle
	pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

	# run kmeans on the coords
	labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
	labels = labels.reshape(-1)  # transpose to row vec

	# segment lines based on their kmeans label
	segmented = defaultdict(list)
	for i, line in enumerate(lines):
		segmented[labels[i]].append(line)
	segmented = list(segmented.values())
	return segmented

def intersection(line1, line2):
	"""Finds the intersection of two lines given in Hesse normal form.

	Returns closest integer pixel locations.
	See https://stackoverflow.com/a/383527/5087436
	"""
	rho1, theta1 = line1[0]
	rho2, theta2 = line2[0]
	A = np.array([
		[np.cos(theta1), np.sin(theta1)],
		[np.cos(theta2), np.sin(theta2)]
	])
	b = np.array([rho1, rho2])
	x0, y0 = np.linalg.solve(A, b)
	x0, y0 = int(np.round(x0)), int(np.round(y0))
	return [x0, y0]

def segmented_intersections(lines):
	"""Finds the intersections between groups of lines."""

	intersections = []
	for i, group in enumerate(lines[:-1]):
		for next_group in lines[i+1:]:
			for line1 in group:
				for line2 in next_group:
					intersections.append(intersection(line1, line2))
	return intersections

def FindTable ( frame: np.ndarray, table_color: np.ndarray, log_images=False ) -> list:
	"""
	table_color: [B, G, R]
	"""

	if log_images:
		LOG_FILE_CNT = 1
		LOG_FILE_DIR = './log_images/'
		LOG_IMG_NAME = lambda name: LOG_FILE_DIR + 'IMG_' + str(LOG_FILE_CNT) + '_' + name + '.jpg'

		cv2.imwrite(LOG_IMG_NAME('origin_frame'), frame)
		LOG_FILE_CNT += 1

	# Filter image by HLS color
	frame_HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
	table_color_HLS = cv2.cvtColor(np.uint8([[table_color]]), cv2.COLOR_BGR2HLS)[0][0]

	HLS_range_size = [15, 70, 50]
	HLS_range_low = table_color_HLS - HLS_range_size
	HLS_range_high = table_color_HLS + HLS_range_size

	filtered = cv2.inRange(frame_HLS, HLS_range_low, HLS_range_high)

	if log_images:
		cv2.imwrite(LOG_IMG_NAME('filtered_result'), filtered)
		LOG_FILE_CNT += 1

	# Image closing
	kernel = np.ones((3,3), np.uint8)
	iters = 2
	filtered = cv2.erode(filtered, kernel, iterations = iters)
	filtered = cv2.dilate(filtered, kernel, iterations = iters)

	if log_images:
		cv2.imwrite(LOG_IMG_NAME('after_closing'), filtered)
		LOG_FILE_CNT += 1

	# Find contour
	contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# filter contours by their area size
	thresh_ratio = 25

	filtered_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > frame.shape[0] * frame.shape[1] / thresh_ratio:
			"""
			epsilon = 0.005 * cv2.arcLength(contour,True)
			approx = cv2.approxPolyDP(contour,epsilon,True)
			"""
			filtered_contours.append(contour)
	
	if log_images:
		copy_frame = copy.copy(frame)
		cv2.drawContours(copy_frame, contours, -1, (0, 255, 0), 1)
		cv2.drawContours(copy_frame, filtered_contours, -1, (0, 0, 255), 2)

		cv2.imwrite(LOG_IMG_NAME('contours'), copy_frame)
		LOG_FILE_CNT += 1

	# Draw contours in binary image

	bin_contour = np.zeros(frame.shape[:2], np.uint8)
	cv2.drawContours(bin_contour, filtered_contours, -1, 255, 1)

	if log_images:
		cv2.imwrite(LOG_IMG_NAME('binary_contours'), bin_contour)
		LOG_FILE_CNT += 1

	# Perform Houph Line Transform
	lines = cv2.HoughLines(image = bin_contour,
							rho = 1,
							theta = 2 * np.pi / 180,
							threshold = 50,
							srn = 0,
							stn = 0)

	if log_images:			
		# Draw the result on origin frame
		copy_frame = copy.copy(frame)
		for line in lines:
			rho = line[0][0]
			theta = line[0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			cv2.line(copy_frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.imwrite(LOG_IMG_NAME('lines_by_houph'), copy_frame)
		LOG_FILE_CNT += 1

	segmented = segment_by_angle_kmeans(lines)

	if log_images:
		copy_frame = copy.copy(frame)
		for (i, group) in enumerate(segmented):
			for line in group:
				rho = line[0][0]
				theta = line[0][1]
				a = math.cos(theta)
				b = math.sin(theta)
				x0 = a * rho
				y0 = b * rho
				pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
				pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
				cv2.line(copy_frame, pt1, pt2, (0, 0, 255) if i == 0 else (0, 255, 0), 2, cv2.LINE_AA)

		cv2.imwrite(LOG_IMG_NAME('segment_by_k-means'), copy_frame)
		LOG_FILE_CNT += 1

	intersections = segmented_intersections(segmented)

	if log_images:
		copy_frame = copy.copy(frame)
		for intersec in intersections:
			cv2.circle(copy_frame, intersec, 3, (0, 0, 255), thickness=-1)

		cv2.imwrite(LOG_IMG_NAME('intersections'), copy_frame)
		LOG_FILE_CNT += 1

	if len(intersections) != 4:
		return None
	return intersections

	


test = cv2.VideoCapture('./resources/edited.mp4')

def getFrame(vidcap, msec):
	vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)
	hasFrames, image = vidcap.read()
	return image

frame = getFrame(test, 1000)
#frame = getFrame(test, 89000)

FindTable(frame, np.array([211, 200, 184]), True)

"""
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
