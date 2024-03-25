import numpy as np
import cv2
import sys
import random
from queue import Queue

class Game():
    def __init__(self, cap):
        self.tcrange = [] # table color range
        self.vidcap = cap
        self.vidlen = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

    def set_tcrange_ff(self, img: np.ndarray, loc: tuple, gap:int=1):
        # Find the table color range with Flood Filled algorithm
        # img: source image
        # loc: [y, x] start position
        assert len(loc) == 2 and len(img.shape) == 3
        h, w = img.shape[:2]
        assert loc[0] >= 0 and loc[0] < w
        assert loc[1] >= 0 and loc[1] < h
        visited = np.zeros(img.shape[:2])
        tcrange = np.array([img[loc],] * 2)
        assert tcrange.shape == (2,3)
        visited[loc] = 1

        que = Queue()
        que.put(loc)
        while not que.empty():
            r, c = que.get()
            between = (img[r][c] >= tcrange[0]-gap).all() and \
                      (img[r][c] <= tcrange[1]+gap).all()
            if not between: continue

            for i in range(3):
                tcrange[0][i] = min(tcrange[0][i], img[r][c][i])
                tcrange[1][i] = max(tcrange[1][i], img[r][c][i])

            if r-1 >= 0 and not visited[r-1][c]:
                visited[r-1][c] = 1
                que.put((r-1, c))
            if r+1 < h and not visited[r+1][c]:
                visited[r+1][c] = 1
                que.put((r+1, c))
            if c-1 >= 0 and not visited[r][c-1]:
                visited[r][c-1] = 1
                que.put((r, c-1))
            if c+1 < h and not visited[r][c+1]:
                visited[r][c+1] = 1
                que.put((r, c+1))
        tcrange[0] -= 5 * gap
        tcrange[1] += 5 * gap
        self.tcrange = tcrange
        print('Set tcrange to:', *tcrange, file=sys.stderr)

    def sep_views(self, sample_rate=1):
        assert 1 <= sample_rate
        samples = (self.vidlen+sample_rate-1) // sample_rate
        cap = self.vidcap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Convert each frame into binary image and calculate the rate of 1s.
        # Find the closest rate
        views = [] # [rate, avg_frame, cnt]
        def calc_rate(f):
            return np.sum(f >= 128)
        def update_view(org, new):
            org[1] = org[1].astype('float64')
            org[1] = (org[1]*org[2] + new[1]) / (org[2]+new[2])
            org[0] = calc_rate(org[1])
            org[2] += 1
            return org

        max_diff = (3 / 100) * ( self.width * self.height )

        cnttt=0
        ret, frame = cap.read()
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bmask = cv2.inRange(frame, *self.tcrange)
            rate = calc_rate(bmask)

            cnttt+=1
            if cnttt%1000==0: 
                print(cnttt, len(views), rate)

            # Binary search for closest rate
            left, right = 0, len(views)
            while left < right:
                mid = (left + right) // 2
                if views[mid][0] >= rate: right = mid
                else: left = mid+1
            cidx = -1 # closest index
            if left >= len(views): cidx = left-1
            elif left-1 < 0: cidx = left
            else: cidx = left - (abs(views[left-1][0]-rate)<abs(views[left][0]-rate))
            
            if cidx == -1 or abs(views[cidx][0]-rate)>max_diff:
                cidx = len(views)
                views.append([rate, bmask, 1])
            else:
                views[cidx] = update_view(views[cidx], [rate, bmask, 1])
            
            # update list views
            while cidx+1 < len(views) and views[cidx+1][0]<views[cidx][0]:
                views[cidx], views[cidx+1] = views[cidx+1], views[cidx]
                cidx+=1
            while cidx-1 >= 0 and views[cidx-1][0]>views[cidx][0]:
                views[cidx], views[cidx-1] = views[cidx-1], views[cidx]
                cidx-=1
            while cidx+1 < len(views) and views[cidx+1][0]-views[cidx][0]<max_diff:
                views[cidx] = update_view(views[cidx], views[cidx+1])
                views.pop(cidx+1)
            while cidx-1 >= 0 and views[cidx][0]-views[cidx-1][0]<max_diff:
                views[cidx] = update_view(views[cidx-1], views[cidx])
                views.pop(cidx)
                cidx -= 1

            for i in range(sample_rate):
                ret, frame = cap.read()

        def filter_view( v ):
            return  v[2] > (2/100) * samples and \
                    v[0] > (2/10) * self.width * self.height
        views = list( filter( filter_view, views) )
        print('Result views:\n', views, file=sys.stderr)
        for v in views:
            cv2.imshow(f'{v[0]}', v[1].astype('uint8'))
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
j = cv2.imread('resources/test.jpg')
j = cv2.cvtColor(j, cv2.COLOR_BGR2HSV)
vidcap = cv2.VideoCapture('./resources/edited.mp4')
g = Game(vidcap)
g.set_tcrange_ff(j, (112, 220))
g.sep_views(50)
"""
# Test by video
vidcap = cv2.VideoCapture('./resources/edited.mp4')
ret, frame = vidcap.read()
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('ffoutput.mp4', fourcc, 20, (width, height))
while ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bin_mask = cv2.inRange(frame, *g.tcrange)
    img = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
    out.write(img)
    ret, frame = vidcap.read()
out.release()
"""