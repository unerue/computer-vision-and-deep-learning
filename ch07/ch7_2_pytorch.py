import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.model(x)
        return x


def training_epoch(dataloader, device, model, loss_fn, optimizer, metric):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        y_onehot = F.one_hot(y, num_classes=10).float()

        y_hat = model(x)
        loss = loss_fn(y_hat, y_onehot)
        acc = metric(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(x)
            print(f'loss: {loss:>7f}, acc: {acc:>7f} [{current:>5d}/{size:>5d}]')


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


def test(dataloader, device, model, metric):
    return validation(dataloader, device, model, metric)


train_data = MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)
train_loader = DataLoader(train_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SequentialModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
metric = Accuracy(task='multiclass', num_classes=10).to(device)

max_epochs = 50
for t in range(max_epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    training_epoch(train_loader, device, model, loss_fn, optimizer, metric)
    val_acc = validation(test_loader, device, model, metric)
    print('val 정확률=', val_acc * 100, '\n')

torch.save(model.state_dict(), 'mnist-net.pth')

model = SequentialModel().to(device)
model.load_state_dict(torch.load('mnist-net.pth'))

test_acc = test(test_loader, device, model, metric)
print('정확률=', test_acc * 100)
