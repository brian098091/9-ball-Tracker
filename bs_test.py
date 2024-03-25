import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import random

def getFrame(vidcap, msec):
	vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)
	hasFrames, image = vidcap.read()
	return image

vidcap = cv2.VideoCapture('./resources/fixed_view_240p.mp4')

ret, frame = vidcap.read()
height, width = frame.shape[:2]
res = []
count = 0

for _ in range(10):
    frameN = random.randint(1, vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.set( cv2.CAP_PROP_POS_FRAMES, frameN )
    ret, frame = vidcap.read()
    res.append(frame)

img = cv2.imread('./resources/table.jpg')
cv2.imshow("origin", img)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for i, col in enumerate(('b', 'g', 'r')):
    histr = cv2.calcHist(cv2.split(img_hls),[i],None,[256],[0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()
exit()
"""
while ret:
    if count%1000==0: print(count)
    res.append(frame)
    ret, frame = vidcap.read()
    count += 1
"""
res = np.array(res)
print(res.shape)
res = np.moveaxis(res, 0, 3)
print(res.shape)
"""
for r in range(0, height, 100):
    for c in range(0, width, 100):
        #r, c = round(height*145/343), round(width*250/610)
        print(r, c)
        tar = res[r][c]
        plt.hist(tar[0], bins=256, range=(0, 255), color="blue")
        plt.savefig('histograms/B/' + str(r) + '_' + str(c) + '.png')
        plt.clf()
        plt.hist(tar[1], bins=256, range=(0, 255), color="green")
        plt.savefig('histograms/G/' + str(r) + '_' + str(c) + '.png')
        plt.clf()
        plt.hist(tar[2], bins=256, range=(0, 255), color="red")
        plt.savefig('histograms/R/' + str(r) + '_' + str(c) + '.png')
"""
"""
for r in range(0, height):
    for c in range(0, width):
        print(r, c)
        mode = scipy.stats.mode(res[r][c], axis=1)
        print(mode)
"""
def color2string(colors):
    return f'{colors[0]} {colors[1]} {colors[2]}'

modes, counts = scipy.stats.mode(res, axis=3)
print(modes.shape, counts.shape)

res_img = frame
#arr = np.empty((height, width), dtype='<U6')
#import pandas as pd 
#df = pd.DataFrame(0, index=np.arange(height), columns=np.arange(width), dtype="string")
print(type(modes), modes.shape)
print(type(frame), frame.shape)
"""
for r in range(0, height):
    for c in range(0, width):
        #rint(modes)
        res_img[r][c] = modes[r][c]#color2string(modes[r][c])
#df = pd.DataFrame(arr)
df.to_csv("./histograms/foo.csv")"""
from PIL import Image
img = Image.fromarray(cv2.cvtColor(modes, cv2.COLOR_BGR2RGB))
#img = scipy.misc.toimage(modes)
cv2.imshow("ester", img)
cv2.waitKey(0)
cv2.destroyAllWindows()