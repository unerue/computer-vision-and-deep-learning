import os
import random

import cv2
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.transforms import ToTensor, PILToTensor


input_dir = "./datasets/oxford_pets/images/images/"
target_dir = "./datasets/oxford_pets/annotations/annotations/trimaps/"
img_siz = (160, 160)  # 모델에 입력되는 영상 크기
n_class = 3  # 분할 레이블 (1:물체, 2:배경, 3:경계)
batch_siz = 32  # 미니 배치 크기

img_paths = sorted(
    [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")]
)
label_paths = sorted(
    [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".png") and not f.startswith(".")
    ]
)


class OxfordPets(Dataset):
    def __init__(self, img_paths, label_paths, train_transform=None, test_transform=None):
        super().__init__()
        self.img_paths = img_paths
        self.label_paths = label_paths
        assert len(self.img_paths) == len(self.label_paths)

        self.train_transform = train_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.train_transform is not None:
            img = self.train_transform(img)

        if self.test_transform is not None:
            label = self.test_transform(label)
            label = label.squeeze(0).type(torch.LongTensor)
            label -= 1  # 부류 번호를 1,2,3에서 0,1,2로 변환

        return img, label


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, padding, bias):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, filters, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            SeparableConv2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.residual = nn.Conv2d(in_channels, filters, 1, stride=2)

    def forward(self, x):
        previous_block_activation = x
        x = self.block(x)
        x += self.residual(previous_block_activation)
        return x


class UpSampling(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.ConvTranspose2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.Upsample(scale_factor=2)
        )
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, filters, 1)
        )

    def forward(self, x):
        previous_block_activation = x
        x = self.block(x)
        x += self.residual(previous_block_activation)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        down_filters = [32, 64, 128, 256]
        self.down_samplings = nn.Sequential()
        for i in range(1, len(down_filters)):
            self.down_samplings.append(DownSampling(down_filters[i-1], down_filters[i]))

        up_filters = [256, 256, 128, 64, 32]
        self.up_samplings = nn.Sequential()
        for i in range(1, len(up_filters)):
            self.up_samplings.append(UpSampling(up_filters[i-1], up_filters[i]))
        
        self.last_conv = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.inc(x)
        x = self.down_samplings(x)
        x = self.up_samplings(x)
        outputs = self.last_conv(x)
        return outputs


def training_epoch(dataloader, device, model, loss_fn, optimizer, metric):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        acc = metric(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(x)
            print(f"loss: {loss:>7f}, acc: {acc:>7f} [{current:>5d}/{size:>5d}]")


def validation(dataloader, device, model, metric):
    acc_list = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            acc = metric(y_hat, y)
            acc_list.append(acc)

    mean_acc = torch.tensor(acc_list).to(device).mean().item()
    return mean_acc


random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)
test_samples = int(len(img_paths) * 0.1)  # 10%를 테스트 집합으로 사용
train_img_paths = img_paths[:-test_samples]
train_label_paths = label_paths[:-test_samples]
test_img_paths = img_paths[-test_samples:]
test_label_paths = label_paths[-test_samples:]

train_transform = transforms.Compose([
    ToTensor(),
    transforms.Resize(img_siz, antialias=True)
])
test_transform = transforms.Compose([
    PILToTensor(),
    transforms.Resize(img_siz, antialias=True)
])
train_data = OxfordPets(
    train_img_paths, 
    train_label_paths, 
    train_transform=train_transform,
    test_transform=test_transform
)
test_data = OxfordPets(
    test_img_paths, 
    test_label_paths, 
    train_transform=train_transform,
    test_transform=test_transform
)
train_loader = DataLoader(train_data, batch_size=batch_siz)
test_loader = DataLoader(test_data, batch_size=batch_siz)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(num_classes=n_class).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
metric = Accuracy(task="multiclass", num_classes=n_class).to(device)

max_epochs = 30
best_acc = 0
for t in range(max_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_epoch(train_loader, device, model, loss_fn, optimizer, metric)
    val_acc = validation(test_loader, device, model, metric)
    print("val 정확률=", val_acc * 100, "\n")
    if val_acc > best_acc:
        torch.save(model.state_dict(), "oxford_seg.pth")
        best_acc = val_acc

model = UNet(num_classes=n_class).to(device)
model.load_state_dict(torch.load("oxford_seg.pth"))

preds = model(test_data[0][0].unsqueeze(0).to(device))
preds = preds.cpu().detach().numpy()
preds = preds.squeeze(0).transpose(1, 2, 0)

cv2.imshow("Sample image", cv2.imread(test_img_paths[0]))  # 0번 영상 디스플레이
cv2.imshow("Segmentation label", cv2.imread(test_label_paths[0]) * 64)
cv2.imshow("Segmentation prediction", preds)  # 0번 영상 예측 결과 디스플레이

cv2.waitKey()
cv2.destroyAllWindows()
