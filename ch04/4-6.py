import skimage
import numpy as np
import cv2
import time

coffee = skimage.data.coffee()

start = time.time()
slic = skimage.segmentation.slic(coffee, compactness=20, n_segments=600, start_label=1)
g = skimage.future.graph.rag_mean_color(coffee, slic, mode="similarity")
ncut = skimage.future.graph.cut_normalized(slic, g)  # 정규화 절단
print(coffee.shape, " Coffee 영상을 분할하는데 ", time.time() - start, "초 소요")

marking = skimage.segmentation.mark_boundaries(coffee, ncut)
ncut_coffee = np.uint8(marking * 255.0)

cv2.imshow("Normalized cut", cv2.cvtColor(ncut_coffee, cv2.COLOR_RGB2BGR))

cv2.waitKey()
cv2.destroyAllWindows()
