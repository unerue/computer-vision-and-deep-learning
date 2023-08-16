import pathlib
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.transforms import ToTensor


class CNNModel(nn.Module):
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


def training_epoch(dataloader, device, model, loss_fn, optimizer, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    acc_list = []
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        acc = metric(y_hat, y)
        acc_list.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(x)
            print(f'loss: {loss:>7f}, acc: {acc:>7f} [{current:>5d}/{size:>5d}]')

    total_loss /= num_batches
    mean_acc = torch.tensor(acc_list).to(device).mean().item()
    return total_loss, mean_acc


def validation(dataloader, device, model, loss_fn, metric):
    num_batches = len(dataloader)
    total_loss = 0
    acc_list = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()
            acc = metric(y_hat, y)
            acc_list.append(acc)

    total_loss /= num_batches
    mean_acc = torch.tensor(acc_list).to(device).mean().item()
    return total_loss, mean_acc


def test(dataloader, device, model, loss_fn, metric):
    _, mean_acc = validation(dataloader, device, model, loss_fn, metric)
    return mean_acc


data_path = pathlib.Path('datasets/stanford_dogs/images/images')

transform = transforms.Compose([
    ToTensor(),
    transforms.Resize((224, 224), antialias=True),
])

ds = ImageFolder(
    data_path,
    transform=transform,
)
test_ds, train_ds = random_split(ds, [0.2, 0.8], generator=torch.Generator().manual_seed(123))
train_loader = DataLoader(train_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CNNModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.000001)
metric = Accuracy(task='multiclass', num_classes=120).to(device)

max_epochs = 200
history = defaultdict(list)
for t in range(max_epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train_loss, train_acc = training_epoch(train_loader, device, cnn, loss_fn, optimizer, metric)
    val_loss, val_acc = validation(test_loader, device, cnn, loss_fn, metric)
    print('val 정확률=', val_acc * 100, '\n')
    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)

torch.save(cnn.state_dict(), 'cnn_for_stanford_dogs.pth')

cnn = CNNModel().to(device)
cnn.load_state_dict(torch.load('cnn_for_stanford_dogs.pth'))

print('정확률=', test(test_loader, device, cnn, loss_fn, metric) * 100)

with open('dog_species_names.txt', 'wb') as f:
    pickle.dump(ds.classes, f)

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.grid()
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'])
plt.grid()
plt.show()