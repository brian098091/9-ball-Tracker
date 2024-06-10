import cv2
import numpy as np
import copy
from log_image import Log
import math

class Frame:
    def __init__(self, no, view, frame_HSV):
        self.frame_no = no
        self.view = view
        self.frame = frame_HSV # HSVVVVVVV

        mask = np.zeros(self.frame.shape, dtype=np.uint8)
        if self.view != None:
            mask = cv2.fillPoly(mask, np.int32([self.view.corners]), [255, 255, 255])
        self.foreground = cv2.bitwise_and(self.frame, mask)
    
    def findBalls(self, ball_dists, log_images=False):
        def filter_ctrs(ctrs, H, W): # Circle detection
            filtered = []
            for ctr in ctrs:
                rot_rect = cv2.minAreaRect(ctr)
                ctrp = cv2.approxPolyDP(ctr, 3, True)
                r1, c1, h, w = cv2.boundingRect(ctrp)
                #w = rot_rect[1][0] # width
                #h = rot_rect[1][1] # height

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

        frame = self.foreground
        H, W = frame.shape[:2]

        if log_images:
            copy_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            points = np.array(self.view.corners, np.int32)
            cv2.polylines(copy_frame,pts=[points],isClosed=True,color=(255,0,255))
            log = Log()
            log.log_image(copy_frame, 'origin frame')
        
        ttt = cv2.cvtColor(frame, cv2. COLOR_HSV2BGR)
        ttt = cv2.cvtColor(ttt, cv2. COLOR_BGR2GRAY)
        edge = cv2.Canny(ttt, 50, 100)
        
        if log_images:
            log.log_image(edge, 'canny')
            cpframe = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        
        hough_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=10, minRadius=5, maxRadius=15)   
        hough_circles = np.uint16(np.around(hough_circles))
        if hough_circles is not None:
            for circle in hough_circles[0, :]:
            # Draw the outer circle
                center_x, center_y, radius = circle
                if log_images:
                    cv2.circle(cpframe, (center_x, center_y), radius, (0, 255, 0), 1)
        if log_images:
            log.log_image(cpframe, 'houph_circle')
                

        # Background substration
        # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bMask = cv2.inRange(frame, *self.view.tcrange)
        bMask = cv2.bitwise_not(bMask)

        if log_images:
            filtered = cv2.bitwise_and(frame, frame, mask=bMask)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)
            log.log_image(filtered, 'background substration')

        # Image closing
        kernel = np.ones((3,3), np.uint8)
        iters = 2
        bMask = cv2.erode(bMask, kernel, iterations = iters)
        bMask = cv2.dilate(bMask, kernel, iterations = iters)

        if log_images:
            log.log_image(filtered, 'after_closing')

        ctrs, hierarchy = cv2.findContours(bMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_ctrs = filter_ctrs(ctrs, *frame.shape[:2])
        if log_images:
            objects = draw_rectangles(ctrs, frame)
            objects = cv2.cvtColor(objects, cv2.COLOR_HSV2BGR)
            log.log_image(objects, 'all_contours')
            #log.save_contour_regions(ctrs, cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))
            
            filteded_objs = draw_rectangles(filtered_ctrs, frame)
            filteded_objs = cv2.cvtColor(filteded_objs, cv2.COLOR_HSV2BGR)
            log.log_image(filteded_objs, 'filtered_contours')
        
        def filter_interesting(ctr):
            rot_rect = cv2.minAreaRect(ctr)
            ctrp = cv2.approxPolyDP(ctr, 3, True)
            r1, c1, w, h = cv2.boundingRect(ctrp)
            return 15/100 > w/W > 1/100 and 15/100 > h/H > 1/100


        interesting_ctrs = list(filter(filter_interesting, ctrs))
        ball_res = [None,] * len(ball_dists) # ((pivot_x, pivot_y), radius)
        for i, ctr in enumerate(interesting_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            roi = frame[y:y+h, x:x+w]
            log.log_image(roi, f'ctr{i}', hsv=1)
            circles = [] # ((pivot_x, pivot_y), radius)
            if max(w,h) / min(w,h) < 1.2: # TODO: size limitation
                r = min(w, h) // 2
                center = (round(x + w/2), round(y + h/2))
                circles.append((center, r))
            else:
                for circle in hough_circles[0, :]:
                    center_x, center_y, radius = circle
                    if 0 <= center_x - x < w and 0 <= center_y - y < h:
                        circles.append(((center_x, center_y), radius))
            
            if len(circles) > 0: print(len(circles), circles[0])
            for circle in circles:
                mask = np.full((H, W), 0, dtype=np.uint8)
                cv2.circle(mask, *circle, 1, -1)
                size = np.sum(mask)
                h = cv2.calcHist([frame],[0],mask,[180],[0, 180]) / size
                s = cv2.calcHist([frame],[1],mask,[256],[0, 256]) / size
                v = cv2.calcHist([frame],[2],mask,[256],[0, 256]) / size
                # print('circle')
                mn_i, mn = -1, math.inf
                for j, dist in enumerate(ball_dists):
                    dh = np.sum(abs(h-dist[0]))
                    ds = np.sum(abs(s-dist[1]))
                    dv = np.sum(abs(v-dist[2]))
                    val = dh+ds+dv
                    if val < mn:
                        mn_i = j
                        mn = val
                if mn < 5:
                    ball_res[mn_i] = circle
        
        # Remaining balls
        for i, ball in enumerate(ball_res):
            if ball != None: continue
            mn = 4
            for circle in hough_circles[0,:]:
                mask = np.full((H, W), 0, dtype=np.uint8)
                cv2.circle(mask, (circle[0], circle[1]), circle[2], 1, -1)
                size = np.sum(mask)
                h = cv2.calcHist([frame],[0],mask,[180],[0, 180]) / size
                s = cv2.calcHist([frame],[1],mask,[256],[0, 256]) / size
                v = cv2.calcHist([frame],[2],mask,[256],[0, 256]) / size
                dh = np.sum(abs(h-ball_dists[i][0]))
                ds = np.sum(abs(s-ball_dists[i][1]))
                dv = np.sum(abs(v-ball_dists[i][2]))
                val = dh+ds+dv
                if val < mn:
                    mn = val
                    ball = ((circle[0], circle[1]), circle[2])

        if log_images:
            cpframe = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            for i, ball in enumerate(ball_res):
                if ball == None: continue
                cv2.circle(cpframe, *ball, (0, 0, 255), 1)
                cv2.putText(cpframe, str(i), ball[0], 
                    cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)
            log.log_image(cpframe, 'res')

        return ball_res
    
    def findBalls_cc(self, hsv_list, log_images=False):
        # hsv_list: each element [i, mn, mx] means
        # the hsv values of ball i is in [mn, mx]
        frame = self.foreground
        if log_images:
            log = Log()
            log.log_image(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR), 'org')

        pivots = np.empty((len(hsv_list), 2))
        for i, (num, mn_hsv, mx_hsv) in enumerate(hsv_list):
            bin = cv2.inRange(frame, mn_hsv, mx_hsv)
            log.log_image(cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR), f'Ball {num}')
            pivots[i][1] = np.ma.array(np.indices(bin.shape)[0], mask=(bin==0)).mean()
            pivots[i][0] = np.ma.array(np.indices(bin.shape)[1], mask=(bin==0)).mean()
            if log_images:
                col = tuple(map(int, mx_hsv//2+mn_hsv//2))
                ff = copy.copy(frame)
                # cv2.circle(frame, tuple(pivots[i].astype(np.int32)), 10, col, 3)
        if log_images:
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            log.log_image(frame, 'f')
        
        # [20, 230, 250]
        # [173, 255, 180]
        bc = np.array([160, 255, 180], dtype=np.uint8)
        diff = abs(frame.astype(np.int32) - bc.astype(np.int32))
        # diff[0] *= 3
        gray = 255 - np.sum(diff, axis=2) // 3
        print(np.max(gray))
        gray = gray.astype(np.uint8)
        log.log_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 'test')

        exit(-1)
            
