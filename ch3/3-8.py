import cv2

img = cv2.imread("rose.png")
patch = img[250:350, 170:270, :]

img = cv2.rectangle(img, (170, 250), (270, 350), (255, 0, 0), 3)
patch1 = cv2.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
patch2 = cv2.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
patch3 = cv2.resize(patch, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

cv2.imshow("Original", img)
cv2.imshow("Resize nearest", patch1)
cv2.imshow("Resize bilinear", patch2)
cv2.imshow("Resize bicubic", patch3)

cv2.waitKey()
cv2.destroyAllWindows()
