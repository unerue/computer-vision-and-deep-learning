from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


torch.set_float32_matmul_precision('medium')


class SequentialModule(L.LightningModule):
    def __init__(self, optimizer_name):
        super().__init__()
        self.reshape = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )
        self.loss = nn.MSELoss()
        self.metric = Accuracy(task='multiclass', num_classes=10)

        self.optimizer_name = optimizer_name
        
        self.train_acc_list = []
        self.val_acc_list = []
        
        self.history = defaultdict(list)

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=0.01)
        elif self.optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x):
        x = self.reshape(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_onehot = F.one_hot(y, num_classes=10).float()

        y_hat = self(x)
        loss = self.loss(y_hat, y_onehot)
        acc = self.metric(y_hat, y)
        self.train_acc_list.append(acc)

        logs = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(logs, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        mean_acc = torch.tensor(self.train_acc_list).mean().item()
        self.history['accuracy'].append(mean_acc)
        self.train_acc_list.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        acc = self.metric(y_hat, y)
        self.val_acc_list.append(acc)

        logs = {'val_acc': acc}
        self.log_dict(logs, prog_bar=True)

        return logs

    def on_validation_epoch_end(self):
        mean_acc = torch.tensor(self.val_acc_list).mean().item()
        self.log('val_acc', mean_acc, prog_bar=True)
        self.history['val_accuracy'].append(mean_acc)
        self.val_acc_list.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        acc = self.metric(y_hat, y)

        logs = {'test_acc': acc}
        self.log_dict(logs, prog_bar=True)


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

model_sgd = SequentialModule('SGD')
trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=50)
trainer.fit(model_sgd, train_dataloaders=train_loader, val_dataloaders=test_loader)

trainer.test(model_sgd, dataloaders=test_loader, ckpt_path='last')

model_adam = SequentialModule('Adam')
trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=50)
trainer.fit(model_adam, train_dataloaders=train_loader, val_dataloaders=test_loader)

trainer.test(model_adam, dataloaders=test_loader, ckpt_path='last')

plt.plot(model_sgd.history['accuracy'], 'r--')
plt.plot(model_sgd.history['val_accuracy'], 'r')
plt.plot(model_adam.history['accuracy'], 'b--')
plt.plot(model_adam.history['val_accuracy'], 'b')
plt.title('Comparison of SGD and Adam optimizers')
plt.ylim((0.7, 1.0))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train_sgd', 'val_sgd', 'train_adam', 'val_adam'])
plt.grid()
plt.show()
