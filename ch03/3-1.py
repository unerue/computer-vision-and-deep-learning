import cv2
import sys

img = cv2.imread("soccer.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv2.imshow("original_RGB", img)
cv2.imshow("Upper left half", img[0 : img.shape[0] // 2, 0 : img.shape[1] // 2, :])
cv2.imshow(
    "Center half",
    img[
        img.shape[0] // 4 : 3 * img.shape[0] // 4,
        img.shape[1] // 4 : 3 * img.shape[1] // 4,
        :,
    ],
)

cv2.imshow("R channel", img[:, :, 2])
cv2.imshow("G channel", img[:, :, 1])
cv2.imshow("B channel", img[:, :, 0])

cv2.waitKey()
cv2.destroyAllWindows()
