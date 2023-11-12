import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('D://chedao2.jpg',0)
#cv.cvtColor(img, cv.COLOR_BGR2GRAY)
Pimg = cv.GaussianBlur(img, (5, 5), 0, 0)
edges = cv.Canny(Pimg,150,300)
def Update():
    plt.subplot(131),plt.imshow(img, cmap= "gray")
    plt.title('Origin Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(lineImg,cmap = 'gray')
    plt.title('Final Image'), plt.xticks([]), plt.yticks([])
    
lines = cv.HoughLinesP(edges, 1.0, np.pi / 180, 20, np.array([]), minLineLength=60, maxLineGap=20)
lineImg = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(img, (x1, y1), (x2, y2), [0, 0, 0], 2)
        cv.line(lineImg, (x1, y1), (x2, y2), [255, 0, 0], 2)
Update()
plt.show()
cv.waitKey(0)
