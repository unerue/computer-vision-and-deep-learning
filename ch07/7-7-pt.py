import winsound

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.model(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SequentialModel().to(device)
model.load_state_dict(torch.load('dmlp_trained.pth'))


def reset():
    global img

    img = np.ones((200, 520, 3), dtype=np.uint8) * 255
    for i in range(5):
        cv2.rectangle(img, (10 + i * 100, 50), (10 + (i + 1) * 100, 150), (0, 0, 255))
    cv2.putText(
        img,
        "e:erase s:show r:recognition q:quit",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        1,
    )


def grab_numerals():
    numerals = []
    for i in range(5):
        roi = img[51:149, 11 + i * 100 : 9 + (i + 1) * 100, 0]
        roi = 255 - cv2.resize(roi, (28, 28), interpolation=cv2.INTER_CUBIC)
        numerals.append(roi)
    numerals = np.array(numerals)
    return numerals


def show():
    numerals = grab_numerals()
    plt.figure(figsize=(25, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(numerals[i], cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def recognition():
    numerals = grab_numerals()
    numerals = numerals.reshape(5, 784)
    numerals = numerals.astype(np.float32)
    numerals = torch.from_numpy(numerals).to(device) / 255.0
    res = model(numerals)  # 신경망 모델로 예측
    res = res.cpu().detach().numpy()
    class_id = np.argmax(res, axis=1)
    for i in range(5):
        cv2.putText(
            img,
            str(class_id[i]),
            (50 + i * 100, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
        )
    winsound.Beep(1000, 500)


BrushSiz = 4
LColor = (0, 0, 0)


def writing(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), BrushSiz, LColor, -1)


reset()
cv2.namedWindow("Writing")
cv2.setMouseCallback("Writing", writing)

while True:
    cv2.imshow("Writing", img)
    key = cv2.waitKey(1)
    if key == ord("e"):
        reset()
    elif key == ord("s"):
        show()
    elif key == ord("r"):
        recognition()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
