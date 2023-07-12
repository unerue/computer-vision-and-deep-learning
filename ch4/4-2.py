import cv2

img = cv2.imread("soccer.jpg")  # 영상 읽기

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny1 = cv2.Canny(gray, 50, 150)  # Tlow=50, Thigh=150으로 설정
canny2 = cv2.Canny(gray, 100, 200)  # Tlow=100, Thigh=200으로 설정

cv2.imshow("Original", gray)
cv2.imshow("Canny1", canny1)
cv2.imshow("Canny2", canny2)

cv2.waitKey()
cv2.destroyAllWindows()
