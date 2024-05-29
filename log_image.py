import sys
import cv2
import os

class Log():
    fcnt = {}
    def __init__(self, folder = None):
        log_file_path = './log_images/'
        if not os.path.isdir(log_file_path):
            os.mkdir(log_file_path)
        if folder == None:
            folder = sys._getframe(1).f_code.co_name
        if folder not in Log.fcnt.keys():
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