import cv2
import sys


img = cv2.imread("soccer.jpg")  # 영상 읽기

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv2.imshow("Image Display", img)  # 윈도우에 영상 표시

cv2.waitKey()
cv2.destroyAllWindows()
