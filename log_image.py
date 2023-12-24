import sys
import cv2
import os

class Log():
    def __init__(self, folder = None):
        log_file_path = './log_images/'
        if not os.path.isdir(log_file_path):
            os.mkdir(log_file_path)
        if folder == None:
            folder = sys._getframe(1).f_code.co_name
        self.path = log_file_path + folder
        self.cnter = 1
        try:
            for f in os.listdir(self.path):
                os.remove(self.path + '/' + f)
        except FileNotFoundError as err:
            os.mkdir(self.path)
    
    def log_image(self, img, name):
        fullname = self.path + '/img' + str(self.cnter) + '_' + name + '.jpg'
        # print(fullname)
        cv2.imwrite(fullname, img)
        self.cnter += 1