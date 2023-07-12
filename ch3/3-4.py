import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("JohnHancocksSignature.png", cv2.IMREAD_UNCHANGED)

t, bin_img = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(bin_img, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0] // 2 : bin_img.shape[0], 0 : bin_img.shape[0] // 2 + 1]
plt.imshow(b, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

se = np.uint8(
    [
        [0, 0, 1, 0, 0],  # 구조 요소
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
)

b_dilation = cv2.dilate(b, se, iterations=1)  # 팽창
plt.imshow(b_dilation, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion = cv2.erode(b, se, iterations=1)  # 침식
plt.imshow(b_erosion, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b_closing = cv2.erode(cv2.dilate(b, se, iterations=1), se, iterations=1)  # 닫기
plt.imshow(b_closing, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()
