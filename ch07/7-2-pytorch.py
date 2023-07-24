from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchmetrics import Accuracy

import lightning as L


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

class SequentialModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential([
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
            nn.Softmax(),
        ])
        self.loss = nn.MSELoss()
        self.metrics = Accuracy(task='multiclass', num_classes=10)

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        preds = self(images)
        loss = self.loss(preds, targets)
        acc = self.metrics(preds, targets).item()
        self.train_acc_list.append(acc)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss, 'train_acc': acc}
    
    def on_train_epoch_end(self):
        train_acc_epoch = torch.tensor(self.train_)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

    def validation_step(self, batch):
        images, targets = batch

        preds = self(images)
        loss = self.loss(preds, targets)
        val_acc = self.val_metrics.compute()
        self.val_metrics.update(preds, targets)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=True, on_epoch=True)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch

        preds = self(images)
        val_acc = self.val_metrics.compute()

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=True, on_epoch=True)

        return {'val_loss': loss, 'val_acc': val_acc}

model = SequentialModel()
trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=50)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

res = model.evaluate(x_test, y_test, verbose=0)
print("정확률=", res[1] * 100)
