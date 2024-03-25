import numpy as np
import cv2
import matplotlib.pyplot as plt

def getFrame(vidcap, msec):
	vidcap.set(cv2.CAP_PROP_POS_MSEC, msec)
	hasFrames, image = vidcap.read()
	return image

vidcap = cv2.VideoCapture('./resources/fixed_view.mp4')

ret, frame = vidcap.read()
frame = getFrame(vidcap, 4000)
height, width = frame.shape[:2]


corners = [(88, 180), (331, 181), (284, 69), (132, 68)]

corners = [(266, 540), (999, 541), (857, 207), (398, 206)]

mask = np.zeros(frame.shape[:-1], dtype=np.uint8)
roi_corners = np.array([corners], dtype=np.int32)

cv2.fillPoly(mask, roi_corners, 255)

# apply the mask
#masked_img = cv2.bitwise_and(frame, mask)
img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#masked_img_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

hist_res = []
for i, channel in enumerate(['h', 's', 'v']):
    maxx = 180 if channel == 'h' else 256
    hist = cv2.calcHist(cv2.split(img_hsv), [i], mask, [maxx], [0, maxx])
    #print(hist.shape)
    hist_res.append(hist)
    plt.plot(hist, color = ['r', 'black', 'gray'][i])
    plt.xlim([0, maxx])

max_counts = [max(r) for r in hist_res]

hue_count = hist_res[0]
hue_table = np.empty(180)
rsize = 10 # range: [h-rsize, h+rsize)
for hue in range(180):
    l, r = max(hue-rsize, 0), min(hue+rsize, 180)
    s = sum(hue_count[l:r])
    x = max(hue_count[l:r])
    print(s, l, r)
    hue_table[hue] = 0
    if hue_count[hue] < max_counts[0]/100: continue
    if s[0]/(r-l) < max_counts[0]/30: continue
    hue_table[hue] = 255
print(hue_table)




foreground_mask = np.zeros(img_hsv.shape, dtype=np.uint8)
for i in range(height):
    for j in range(width):
        #if mask[i][j] > 0 and hist_res[0][img_hsv[i][j][0]] < max_counts[0][0]/100:
        if mask[i][j] > 0 and hue_table[img_hsv[i][j][0]] == 0:
            foreground_mask[i][j] = (255,)*3
print(img_hsv.shape, foreground_mask.shape)
# cv2.imshow('forem', foreground_mask)
masked_img = cv2.bitwise_and(frame, foreground_mask)


plt.show()


cv2.imshow('original frame', frame)
cv2.imshow('masked', masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()