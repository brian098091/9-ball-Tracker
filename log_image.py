import sys
import cv2
import os
import numpy as np

class Log():
    fcnt = {}
    def __init__(self, folder = None):
        log_file_path = './log_images/'
        if not os.path.isdir(log_file_path):
            os.mkdir(log_file_path)
        if folder == None:
            folder = sys._getframe(1).f_code.co_name
        if folder not in Log.fcnt:
            Log.fcnt[folder] = 0
        self.folder = folder
        self.path = log_file_path + folder
        self.cnter = 1
        try:
            for f in os.listdir(self.path):
                if Log.fcnt[folder] == 0:
                    os.remove(self.path + '/' + f)
        except FileNotFoundError as err:
            os.mkdir(self.path)
        Log.fcnt[folder] += 1
    
    def log_image(self, img, name):
        fullname = f'{self.path}/c{str(Log.fcnt[self.folder])}_img{str(self.cnter)}_{name}.jpg'
        # print(fullname)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(fullname, img)
        self.cnter += 1
    
    def save_contour_regions(self,ctrs, img, prefix='contour'):
        for i,ctr in enumerate(ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            roi = img[y:y+h, x:x+w]
            if w > 30:
                cv2.imwrite(f'./test_crts/c{str(Log.fcnt[self.folder])}_contour{i}.jpg',roi)
                GrayImage = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                GrayImage = cv2.GaussianBlur(GrayImage, (9, 9), 2)

                th = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                

                kernel = np.ones((5, 5), np.uint8)
                erosion = cv2.erode(th, kernel, iterations=1)
                dilation = cv2.dilate(erosion, kernel, iterations=1)
                imgray = cv2.Canny(dilation, 10, 70)

                circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20,
                                           param1=50, param2=10, minRadius=5, maxRadius=15)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                    # Draw the outer circle
                        center_x, center_y, radius = circle
                        original_center_x = x + center_x
                        original_center_y = y + center_y
                        cv2.circle(img, (original_center_x, original_center_y), radius, (0, 255, 0), 2)
                cv2.imwrite(f'./test_crts/c{str(Log.fcnt[self.folder])}_balls.jpg',img)
        
        
        