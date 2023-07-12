import cv2
import numpy as np

img = cv2.imread("soccer.jpg")
img = cv2.resize(img, dsize=(0, 0), fx=0.4, fy=0.4)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.putText(gray, "soccer", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow("Original", gray)

smooth = np.hstack(
    (
        cv2.GaussianBlur(gray, (5, 5), 0.0),
        cv2.GaussianBlur(gray, (9, 9), 0.0),
        cv2.GaussianBlur(gray, (15, 15), 0.0),
    )
)
cv2.imshow("Smooth", smooth)

femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv2.filter2D(gray16, -1, femboss) + 128, 0, 255))
emboss_bad = np.uint8(cv2.filter2D(gray16, -1, femboss) + 128)
emboss_worse = cv2.filter2D(gray, -1, femboss)

cv2.imshow("Emboss", emboss)
cv2.imshow("Emboss_bad", emboss_bad)
cv2.imshow("Emboss_worse", emboss_worse)

cv2.waitKey()
cv2.destroyAllWindows()
