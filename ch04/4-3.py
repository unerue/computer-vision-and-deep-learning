import cv2
import numpy as np

img = cv2.imread("soccer.jpg")  # 영상 읽기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

contour, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

lcontour = []
for i in range(len(contour)):
    if contour[i].shape[0] > 100:  # 길이가 100보다 크면
        lcontour.append(contour[i])

cv2.drawContours(img, lcontour, -1, (0, 255, 0), 3)

cv2.imshow("Original with contours", img)
cv2.imshow("Canny", canny)

cv2.waitKey()
cv2.destroyAllWindows()
