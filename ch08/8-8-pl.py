import pickle
import sys
# import winsound

import cv2
import numpy as np
import lightning as L
import torch
from PyQt5.QtWidgets import *
from torch import nn
from torchvision import transforms
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.transforms import ToTensor


torch.set_float32_matmul_precision('medium')


class CNNModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.model = nn.Sequential(
            base_model.features,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.75),
            nn.Linear(1024, 120),
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CNNModule.load_from_checkpoint('cnn_for_stanford_dogs.ckpt').to(device)  # 모델 읽기
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))  # 견종 이름


class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('견종 인식')
        self.setGeometry(200, 200, 700, 100)

        fileButton = QPushButton('강아지 사진 열기', self)
        recognitionButton = QPushButton('품종 인식', self)
        quitButton = QPushButton('나가기', self)

        fileButton.setGeometry(10, 10, 100, 30)
        recognitionButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, '강아지 사진 읽기', './')
        self.img = cv2.imread(fname[0])
        if self.img is None:
            sys.exit('파일을 찾을 수 없습니다.')

        cv2.imshow('Dog image', self.img)

    def recognitionFunction(self):
        transform = transforms.Compose([
            ToTensor(),
            transforms.Resize((224, 224), antialias=True),
        ])
        x = transform(self.img)
        x = x.to(device)
        print(x.shape)
        res = cnn(x)  # 예측
        res = res.cpu().detach().numpy()
        top5 = np.argsort(-res)[:5]
        top5_dog_species_names = [dog_species[i] for i in top5]
        for i in range(5):
            prob = '(' + str(res[top5[i]]) + ')'
            name = str(top5_dog_species_names[i]).split('-')[1]
            cv2.putText(
                self.img,
                prob + name,
                (10, 100 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        cv2.imshow('Dog image', self.img)
        # winsound.Beep(1000, 500)

    def quitFunction(self):
        cv2.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()
