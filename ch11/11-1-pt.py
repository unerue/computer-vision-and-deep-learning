import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


train_data = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
train_loader = DataLoader(train_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)

n_class = 10  # 부류 수
img_siz = (32, 32, 3)  # 영상의 크기

patch_siz = 4  # 패치 크기
p2 = (img_siz[0] // patch_siz) ** 2  # 패치 개수
d_model = 64  # 임베딩 벡터 차원
h = 8  # 헤드 개수
N = 6  # 인코더 블록의 개수


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p_siz = patch_size
        self.extract_patches = nn.Unfold(self.p_siz, stride=self.p_siz)

    def forward(self, img):
        batch_size = img.shape[0]
        patches = self.extract_patches(img).permute(0, 2, 1)
        patch_dims = patches.shape[-1]