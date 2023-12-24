import cv2
import numpy as np
import copy
from table import Table
from log_image import Log

def FindBalls( frame: np.ndarray, table: Table, log_images=False ):
    if log_images:
        copy_frame = copy.copy(frame)
        points = np.array(table.corners, np.int32)
        cv2.polylines(copy_frame,pts=[points],isClosed=True,color=(255,0,255))
        log = Log()
        log.log_image(copy_frame, 'origin frame')

    # Background substration
    frame_HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    filtered = cv2.inRange(frame_HLS, table.color_min, table.color_max)
    filtered = cv2.bitwise_not(filtered)

    if log_images: log.log_image(filtered, 'background substration')

    # Image closing
    kernel = np.ones((3,3), np.uint8)
    iters = 2
    filtered = cv2.erode(filtered, kernel, iterations = iters)
    filtered = cv2.dilate(filtered, kernel, iterations = iters)

    if log_images:
        log.log_image(filtered, 'after_closing')
