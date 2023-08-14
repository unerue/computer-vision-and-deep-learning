from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(3072, 1024),
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


def training_epoch(dataloader, device, model, loss_fn, optimizer, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    acc_list = []
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        y_onehot = F.one_hot(y, num_classes=10).float()

        y_hat = model(x)
        loss = loss_fn(y_hat, y_onehot)
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


def validation(dataloader, device, model, metric):
    num_batches = len(dataloader)
    total_loss = 0
    acc_list = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_onehot = F.one_hot(y, num_classes=10).float()

            y_hat = model(x)
            loss = loss_fn(y_hat, y_onehot)
            total_loss += loss.item()
            acc = metric(y_hat, y)
            acc_list.append(acc)

    total_loss /= num_batches
    mean_acc = torch.tensor(acc_list).to(device).mean().item()
    return total_loss, mean_acc


def test(dataloader, device, model, metric):
    _, mean_acc = validation(dataloader, device, model, metric)
    return mean_acc


train_data = CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
train_loader = DataLoader(train_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dmlp = SequentialModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(dmlp.parameters(), lr=0.0001)
metric = Accuracy(task='multiclass', num_classes=10).to(device)

max_epochs = 50
history = defaultdict(list)
for t in range(max_epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train_loss, train_acc = training_epoch(train_loader, device, dmlp, loss_fn, optimizer, metric)
    val_loss, val_acc = validation(test_loader, device, dmlp, metric)
    print('val 정확률=', val_acc * 100, '\n')
    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)

torch.save(dmlp.state_dict(), 'dmlp_trained.pth')

dmlp = SequentialModel().to(device)
dmlp.load_state_dict(torch.load('dmlp_trained.pth'))

print('정확률=', test(test_loader, device, dmlp, metric) * 100)

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
