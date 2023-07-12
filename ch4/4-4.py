import cv2

img = cv2.imread("apples.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

apples = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120
)

for i in apples[0]:
    cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

cv2.imshow("Apple detection", img)

cv2.waitKey()
cv2.destroyAllWindows()
