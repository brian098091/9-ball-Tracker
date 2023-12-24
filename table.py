import numpy as np

class Table():

    def __init__(self, color=None):
        self.color_min = None
        self.color_max = None
        self.corners = None
        self.balls = None
        if color != None: self.set_hls_color(color)

    def set_hls_color( self, hls_color: np.ndarray, hls_color_range=np.asarray([15, 70, 50]) ):
        if len(hls_color) != 3 or len(hls_color_range) != 3:
            return -1
        self.color_min = hls_color - hls_color_range
        self.color_max = hls_color + hls_color_range
        for i in range(3):
            if self.color_min[i] < 0: self.color_min[i] = 0
            if self.color_max[i] < 0: self.color_max[i] = 0
            if self.color_min[i] > 255: self.color_min[i] = 255
            if self.color_max[i] > 255: self.color_max[i] = 255
        return 0