import skimage
import numpy as np
import cv2

img = skimage.data.coffee()
cv2.imshow("Coffee image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

slic1 = skimage.segmentation.slic(img, compactness=20, n_segments=600)
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)
sp_img1 = np.uint8(sp_img1 * 255.0)

slic2 = skimage.segmentation.slic(img, compactness=40, n_segments=600)
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2 * 255.0)

cv2.imshow("Super pixels (compact 20)", cv2.cvtColor(sp_img1, cv2.COLOR_RGB2BGR))
cv2.imshow("Super pixels (compact 40)", cv2.cvtColor(sp_img2, cv2.COLOR_RGB2BGR))

cv2.waitKey()
cv2.destroyAllWindows()
