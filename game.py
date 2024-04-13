import numpy as np
import cv2
import sys
import random
from view import View
from queue import Queue

class Game():
    def __init__(self, cap, sample_rate=1):
        self.vidcap = cap
        self.sample_rate = sample_rate
        self.vidlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tcrange = [] # table color range
        self.frame_count = (self.vidlen - 1) // sample_rate + 1
        self.frames = np.empty(self.frame_count, dtype=object)


    def set_tcrange_ff(self, 
        img: np.ndarray, 
        loc: tuple, 
        gap: np.ndarray = np.array([1,1,1], dtype='uint8')
        ):
        # Find the table color range with Flood Filled algorithm
        # img: source image
        # loc: [y, x] start position
        assert len(loc) == 2 and len(img.shape) == 3
        h, w = img.shape[:2]
        assert loc[0] >= 0 and loc[0] < h
        assert loc[1] >= 0 and loc[1] < w
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
        tcrange[0] -= np.minimum(tcrange[0], 5 * gap)
        tcrange[1] += np.minimum(255-tcrange[1], 5 * gap)
        self.tcrange = tcrange
        print('Set tcrange to:', *tcrange, file=sys.stderr)
        return cv2.inRange(img, *tcrange)

    def sep_views(self, diff_percentage=3):
        assert 1 <= self.sample_rate
        samples = self.frame_count

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

        max_diff = (diff_percentage / 100) * ( self.width * self.height )
        idx = 0
        ret, frame = cap.read()
        rates = np.empty(self.frame_count, dtype='int32')
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bmask = cv2.inRange(frame, *self.tcrange)
            rate = calc_rate(bmask)
            rates[idx] = rate

            # Update views
            # Binary search for closest rate
            left, right = 0, len(views)
            while left < right:
                mid = (left + right) // 2
                if views[mid][0] >= rate: right = mid
                else: left = mid+1
            cidx = -1 # closest index
            if left >= len(views): cidx = left-1
            elif left-1 < 0: cidx = left
            else: cidx = left - ( abs(views[left-1][0]-rate)<abs(views[left][0]-rate) )
            
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

            for i in range(self.sample_rate):
                ret, frame = cap.read()

            idx += 1
            if idx % 100==0: 
                print(idx, len(views), rate)

        def filter_view( v ):
            return  v[2] > (2/100) * samples and \
                    v[0] > (2/10) * self.width * self.height
        views = list( filter( filter_view, views) )
        view_objs = []
        print('Result views:\n', views, file=sys.stderr)
        for i, v in enumerate(views):
            cv2.imwrite(f'./tmpOutput/view_{i}.jpg', v[1].astype('uint8'))
            view_objs.append(View(i, v[1]))

        return
        for i, r in enumerate(rates):
            for j, v in enumerate(views):
                if abs(r - v[0]) <= max_diff:
                    self.frames[i] = Frame(i, view_obj[j])
                    break
                    
        
        
if __name__ == '__main__':
    j = cv2.imread('resources/2022_APP_2_000.jpg')
    j = cv2.cvtColor(j, cv2.COLOR_BGR2HSV)
    vidcap = cv2.VideoCapture('./resources/2022_APP_2.mp4')
    g = Game(vidcap, 4)
    ret=g.set_tcrange_ff(j, (520, 520), gap=np.array([10,15,15], dtype='uint8')) # (112, 220)
    print('Please wait, 3Q')
    # Test by video

    vidcap = cv2.VideoCapture('./resources/2022_APP_2.mp4')
    ret, frame = vidcap.read()
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./tmpOutput/ffoutput.mp4', fourcc, 20, (width, height))
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bin_mask = cv2.inRange(frame, *g.tcrange)
        img = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
        out.write(img)
        ret, frame = vidcap.read()
    out.release()

    # Test background substraction by video
    vidcap = cv2.VideoCapture('./resources/2022_APP_2.mp4')
    ret, frame = vidcap.read()
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./tmpOutput/background_subtraction.mp4', fourcc, 20, (width, height))
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bin_mask = cv2.inRange(frame, *g.tcrange)
        res = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(bin_mask))
        # res[np.where(res == [0, 0, 0])] = [230, 100, 100]

        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)

        m = np.all(res[:, :, :3] == [0,0,0], axis=-1)
        res[m, :3] = [117, 117, 117]

        out.write(res)
        ret, frame = vidcap.read()
    out.release()

    g.sep_views()