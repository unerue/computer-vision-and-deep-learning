import cv2
import matplotlib.pyplot as plt

img = cv2.imread("mistyroad.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 명암 영상으로 변환하고 출력
plt.imshow(gray, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.show()

h = cv2.calcHist([gray], [0], None, [256], [0, 256])  # 히스토그램을 구해 출력
plt.plot(h, color="r", linewidth=1), plt.show()

equal = cv2.equalizeHist(gray)  # 히스토그램을 평활화하고 출력
plt.imshow(equal, cmap="gray"), plt.xticks([]), plt.yticks([]), plt.show()

h = cv2.calcHist([equal], [0], None, [256], [0, 256])  # 히스토그램을 구해 출력
plt.plot(h, color="r", linewidth=1), plt.show()
