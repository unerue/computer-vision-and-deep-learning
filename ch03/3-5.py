import cv2
import numpy as np

img = cv2.imread("soccer.jpg")
img = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)


def gamma(f, gamma=1.0):
    f1 = f / 255.0  # L=256이라고 가정
    return np.uint8(255 * (f1**gamma))


gc = np.hstack(
    (
        gamma(img, 0.5),
        gamma(img, 0.75),
        gamma(img, 1.0),
        gamma(img, 2.0),
        gamma(img, 3.0),
    )
)
cv2.imshow("gamma", gc)

cv2.waitKey()
cv2.destroyAllWindows()
