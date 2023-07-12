import cv2

img = cv2.imread("mot_color70.jpg")  # 영상 읽기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

gray = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("sift", gray)

k = cv2.waitKey()
cv2.destroyAllWindows()
