import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)

model = ResNet50(weights="imagenet")

img = cv2.imread("rabbit.jpg")
x = np.reshape(cv2.resize(img, (224, 224)), (1, 224, 224, 3))
x = preprocess_input(x)

preds = model.predict(x)
top5 = decode_predictions(preds, top=5)[0]
print("예측 결과:", top5)

for i in range(5):
    cv2.putText(
        img,
        top5[i][1] + ":" + str(top5[i][2]),
        (10, 20 + i * 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

cv2.imshow("Recognition result", img)

cv2.waitKey()
cv2.destroyAllWindows()
