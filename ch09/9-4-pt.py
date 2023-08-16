import numpy as np
import os
import random
import cv2
from torch.utils.data import Dataset


input_dir = "./datasets/oxford_pets/images/images/"
target_dir = "./datasets/oxford_pets/annotations/annotations/trimaps/"
img_siz = (160, 160)  # 모델에 입력되는 영상 크기
n_class = 3  # 분할 레이블 (1:물체, 2:배경, 3:경계)
batch_siz = 32  # 미니 배치 크기


class OxfordPets(Dataset):
    def __init__(self, ):
        super().__init__()