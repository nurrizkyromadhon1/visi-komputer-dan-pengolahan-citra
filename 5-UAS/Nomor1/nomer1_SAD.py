import cv2
import numpy as np
import sys

def template_matching_sad(src, temp):
    h, w = src.shape
    ht, wt = temp.shape

    score = np.empty((h-ht, w-wt))

    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            diff = np.abs(src[dy:dy + ht, dx:dx + wt] - temp)
            score[dy, dx] = diff.sum()

    pt = np.unravel_index(score.argmin(), score.shape)

    return (pt[1], pt[0])


def main():
    img = cv2.imread(sys.path[0]+"/Lenna.png")
    temp = cv2.imread(sys.path[0]+"/Lenna_mata.png")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    h, w = temp.shape

    pt = template_matching_sad(gray, temp)

    #match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF_NORMED)
    #min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    #pt = min_pt


    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0,0,200), 3)
    cv2.imshow('SAD', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()