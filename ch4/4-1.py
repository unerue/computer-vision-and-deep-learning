import cv2

img = cv2.imread("soccer.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # 소벨 연산자 적용
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(grad_x)  # 절대값을 취해 양수 영상으로 변환
sobel_y = cv2.convertScaleAbs(grad_y)

edge_strength = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # 에지 강도 계산

cv2.imshow("Original", gray)
cv2.imshow("sobelx", sobel_x)
cv2.imshow("sobely", sobel_y)
cv2.imshow("edge strength", edge_strength)

cv2.waitKey()
cv2.destroyAllWindows()
