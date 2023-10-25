import cv2
import numpy as np
import math
import copy

def FindTable ( frame: np.ndarray, table_color: np.ndarray ) -> list:
	"""
	table_color: [B, G, R]
	"""

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

	cv2.imwrite(LOG_IMG_NAME('filtered_result'), filtered)
	LOG_FILE_CNT += 1

	# Image closing
	kernel = np.ones((3,3), np.uint8)
	iters = 2
	filtered = cv2.erode(filtered, kernel, iterations = iters)
	filtered = cv2.dilate(filtered, kernel, iterations = iters)

	cv2.imwrite(LOG_IMG_NAME('after_closing'), filtered)
	LOG_FILE_CNT += 1

	# Find contour
	copy_frame = copy.copy(frame)
	contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(copy_frame, contours, -1, (0, 255, 0), 1)

	# filter contours by their area size
	thresh_ratio = 25

	filtered_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > frame.shape[0] * frame.shape[1] / thresh_ratio:
			filtered_contours.append(contour)
	
	cv2.drawContours(copy_frame, filtered_contours, -1, (0, 0, 255), 1)

	cv2.imwrite(LOG_IMG_NAME('contours'), copy_frame)
	LOG_FILE_CNT += 1

	# Draw contours in binary image

	bin_contour = np.zeros(frame.shape[:2], np.uint8)
	cv2.drawContours(bin_contour, filtered_contours, -1, 255, 1)

	cv2.imwrite(LOG_IMG_NAME('contours'), bin_contour)
	LOG_FILE_CNT += 1

	# Perform Houph Line Transform
	copy_frame = copy.copy(frame)
	lines = cv2.HoughLines(image = bin_contour,
							rho = 5,
							theta = 10 * np.pi / 180,
							threshold = 75,
							srn=0,
							stn=0)
	for line in lines:
		rho = line[0][0]
		theta = line[0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = a * rho
		y0 = b * rho
		pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
		pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
		cv2.line(copy_frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

	cv2.imwrite(LOG_IMG_NAME('lines_by_houph'), copy_frame)
	LOG_FILE_CNT += 1
	


test = cv2.VideoCapture('./resources/edited.mp4')

def getFrame(vidcap, msec):
	vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)
	hasFrames, image = vidcap.read()
	return image

frame = getFrame(test, 1000)
frame = getFrame(test, 89000)

# cv2.imwrite("frame1.jpg", frame)



FindTable(frame, np.array([211, 200, 184]))



# cv2.imshow("TEST", frame)

"""
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
