import cv2
import numpy as np
import math
import copy
import os
import random
from log_image import Log
from collections import defaultdict

class View:
    def __init__(self, no, tcrange, avg_mask):
        self.view_no = no
        self.avg_mask = avg_mask
        self.tcrange = tcrange.astype('uint8')
        self.corners = self.find_baize(self.avg_mask, True)
        assert self.corners != None
        self.corners = np.array(self.corners, dtype=np.int32)
    
    def find_baize(self, mask: np.ndarray, log_images=False):
        
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

        def findCorner(bounds):
            """Finds the intersections between top, bottom, left, right."""

            intersections = []
            intersections.append(intersection(bounds[0], bounds[2]))
            intersections.append(intersection(bounds[0], bounds[3]))
            intersections.append(intersection(bounds[1], bounds[3]))
            intersections.append(intersection(bounds[1], bounds[2]))
            return intersections

        if log_images:
            log = Log()
            log.log_image(mask, 'origin_frame')
        
        # filtered = cv2.inRange(frame, *self.tcrange)
        filtered = mask
        if log_images:
            log.log_image(filtered, 'filtered_result')

        # Image closing
        kernel = np.ones((3,3), np.uint8)
        iters = 2
        filtered = cv2.erode(filtered, kernel, iterations = iters)
        filtered = cv2.dilate(filtered, kernel, iterations = iters)

        if log_images:
            log.log_image(filtered, 'after_closing')

        # Find contour
        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # filter contours by their area size
        thresh_ratio = 25

        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > filtered.shape[0] * filtered.shape[1] / thresh_ratio:
                filtered_contours.append(contour)
        if len(filtered_contours) == 0: return None
        
        if log_images:
            copy_frame = copy.copy(mask)
            cv2.drawContours(copy_frame, contours, -1, (0, 255, 0), 1)
            cv2.drawContours(copy_frame, filtered_contours, -1, (0, 0, 255), 2)

            log.log_image(copy_frame, 'contours')

        # Draw contours in binary image
        bin_contour = np.zeros(filtered.shape[:2], np.uint8)
        cv2.drawContours(bin_contour, filtered_contours, -1, 255, 2)

        bin_contour = cv2.Canny(mask, 100, 200)

        if log_images:
            log.log_image(bin_contour, 'binary_contours')

        def is_bound(bound_num, line):
            """
            bound_num: 0(top), 1(bottom), 2(left), 3(right)
            """
            if math.radians(60) < line[0][1] < math.radians(120):
                # horizontal line
                if line[0][0] < filtered.shape[0] / 2:
                    return bound_num == 0
                else:
                    return bound_num == 1
            else:
                # vertical line
                if abs(line[0][0]) < filtered.shape[1] / 2:
                    return bound_num == 2
                else:
                    return bound_num == 3

        # Perform Houph Line Transform
        bounds = []
        for bound_num in range(4):
            done = False
            min_thresh = 50
            max_thresh = 500
            last_res = None
            while not done:
                if min_thresh > max_thresh:
                    # No boundary found
                    break
                thresh = min_thresh + (max_thresh - min_thresh) // 2
                lines = cv2.HoughLines(image = bin_contour,
                                        rho = 1,
                                        theta = 2 * np.pi / 180,
                                        threshold = thresh,
                                        srn = 0,
                                        stn = 0)
                
                if log_images:
                    ccopy_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    for line in lines:
                        rho = line[0][0]
                        theta = line[0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 3000*(-b)), int(y0 + 3000*(a)))
                        pt2 = (int(x0 - 3000*(-b)), int(y0 - 3000*(a)))
                        cv2.line(ccopy_frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

                    log.log_image(ccopy_frame, 'changing threshold')

                #print(lines)
                res = []
                if lines is not None:
                    for line in lines:
                        if is_bound(bound_num, line):
                            res.append(line)
                print(thresh, len(res))
                if len(res) > 0:
                    last_res = res
                if len(res) < 1:
                    max_thresh = thresh - 1
                elif len(res) > 1:
                    min_thresh = thresh + 1
                else:
                    bounds.append(res[0])
                    done = True
            if not done:
                bounds.append(last_res[0])

            if not done:print('Not Done!')
        if log_images:
            copy_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for line in bounds:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 3000*(-b)), int(y0 + 3000*(a)))
                pt2 = (int(x0 - 3000*(-b)), int(y0 - 3000*(a)))
                cv2.line(copy_frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

            log.log_image(copy_frame, 'filtered four line')

        intersections = findCorner(bounds)

        if log_images:
            copy_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for intersec in intersections:
                cv2.circle(copy_frame, intersec, 3, (0, 0, 255), thickness=-1)

            log.log_image(copy_frame, 'intersections')

        if len(intersections) != 4:
            return None
        return intersections