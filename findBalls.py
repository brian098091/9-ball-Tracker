import cv2
import numpy as np
import copy
from log_image import Log

def filter_ctrs(ctrs, W, H):
    filtered = []
    for ctr in ctrs:
        rot_rect = cv2.minAreaRect(ctr)
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height

        if max(w/h, h/w) >= 2: continue
        if w/W < 1/100 or w/W > 10/100: continue
        if h/H < 1/100 or h/H > 10/100: continue

        filtered.append(ctr)
    return filtered

def draw_rectangles(ctrs, img):
    output = img.copy()

    for i in range(len(ctrs)):
        M = cv2.moments(ctrs[i]) # moments
        rot_rect = cv2.minAreaRect(ctrs[i])
        w = rot_rect[1][0] # width
        h = rot_rect[1][1] # height
        
        box = np.int64(cv2.boxPoints(rot_rect))
        cv2.drawContours(output,[box],0,(0,0,255),3) # draws box
        
    return output

def FindBalls( frame: np.ndarray, tcrange: np.ndarray, log_images=False ):
    if log_images:
        copy_frame = copy.copy(frame)
        points = np.array(table.corners, np.int32)
        cv2.polylines(copy_frame,pts=[points],isClosed=True,color=(255,0,255))
        log = Log()
        log.log_image(copy_frame, 'origin frame')

    # Background substration
    # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bMask = cv2.inRange(frame, *tcrange)
    bMask = cv2.bitwise_not(bMask)

    if log_images: log.log_image(filtered, 'background substration')

    # Image closing
    kernel = np.ones((3,3), np.uint8)
    iters = 2
    bMask = cv2.erode(bMask, kernel, iterations = iters)
    bMask = cv2.dilate(bMask, kernel, iterations = iters)

    if log_images:
        log.log_image(filtered, 'after_closing')

    ctrs, hierarchy = cv2.findContours(bMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = filter_ctrs(ctrs, *frame.shape[:2])
    #detected_objects_filtered = draw_rectangles(ctrs_filtered, frame)
    if log_images:
        detected_objects = draw_rectangles(ctrs, frame) 
        log.log_image(detected_objects, 'contours')
        #log.log_image(detected_objects_filtered, 'filtered contours')
    return ctrs
