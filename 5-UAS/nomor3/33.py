import cv2
import numpy as np

img = cv2.imread("./bola2.jpg")
(h,w,c) = img.shape
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

parts = []
step_x = 3
step_y = 3
eqs = []
eq_img = np.zeros_like(gray_img)

for x in range(step_x):
    for y in range(step_y):
        xratio1 = x/step_x
        xratio2 = (x+1)/step_x
        yratio1 = y/step_y
        yratio2 = (y+1)/step_y
        part = gray_img[int(yratio1*h):int(yratio2*h),int(xratio1*w):int(xratio2*w)].copy()
        parts.append(part)

        cv2.imshow("x = {0}, y = {1}".format(x,y),part)

        eq = cv2.equalizeHist(part)
        eqs.append(eq)
        eq_img[int(yratio1*h):int(yratio2*h),int(xratio1*w):int(xratio2*w)] = eq

cv2.imshow("eq_img",eq_img)
cv2.waitKey(0)