import pathlib
import pickle
from collections import defaultdict

import lightning as L
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.densenet import DenseNet121_Weights, densenet121
from torchvision.transforms import ToTensor


torch.set_float32_matmul_precision("medium")


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
        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy(task="multiclass", num_classes=120)
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

        self.history = defaultdict(list)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.000001)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.metric(y_hat, y)
        self.train_loss_list.append(loss)
        self.train_acc_list.append(acc)

        logs = {"train_loss": loss, "train_acc": acc}
        self.log_dict(logs, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        mean_loss = torch.tensor(self.train_loss_list).mean().item()
        mean_acc = torch.tensor(self.train_acc_list).mean().item()
        self.history["loss"].append(mean_loss)
        self.history["accuracy"].append(mean_acc)
        self.train_loss_list.clear()
        self.train_acc_list.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.metric(y_hat, y)
        self.val_loss_list.append(loss)
        self.val_acc_list.append(acc)

        logs = {"val_acc": acc}
        self.log_dict(logs, prog_bar=True)

        return logs

    def on_validation_epoch_end(self):
        mean_loss = torch.tensor(self.val_loss_list).mean().item()
        mean_acc = torch.tensor(self.val_acc_list).mean().item()
        self.log("val_acc", mean_acc, prog_bar=True)
        self.history["val_loss"].append(mean_loss)
        self.history["val_accuracy"].append(mean_acc)
        self.val_loss_list.clear()
        self.val_acc_list.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        acc = self.metric(y_hat, y)

        logs = {"test_acc": acc}
        self.log_dict(logs, prog_bar=True)


data_path = pathlib.Path("datasets/stanford_dogs/images/images")

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

cnn = CNNModule()
trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=200)
trainer.fit(cnn, train_dataloaders=train_loader, val_dataloaders=test_loader)

trainer.save_checkpoint("cnn_for_stanford_dogs.ckpt")
trainer.test(cnn, dataloaders=test_loader, ckpt_path="cnn_for_stanford_dogs.ckpt")

# trainer.test(cnn, dataloaders=test_loader, ckpt_path="last")

with open("dog_species_names.txt", "wb") as f:
    pickle.dump(ds.classes, f)

plt.plot(cnn.history["accuracy"])
plt.plot(cnn.history["val_accuracy"])
plt.title("Accuracy graph")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["train", "test"])
plt.grid()
plt.show()

plt.plot(cnn.history["loss"])
plt.plot(cnn.history["val_loss"])
plt.title("Loss graph")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train", "test"])
plt.grid()
plt.show()
