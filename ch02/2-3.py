import cv2
import sys


img = cv2.imread("soccer.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환
gray_small = cv2.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)  # 반으로 축소

cv2.imwrite("soccer_gray.jpg", gray)  # 영상을 파일에 저장
cv2.imwrite("soccer_gray_small.jpg", gray_small)

cv2.imshow("Color image", img)
cv2.imshow("Gray image", gray)
cv2.imshow("Gray image small", gray_small)

cv2.waitKey()
cv2.destroyAllWindows()
