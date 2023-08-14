import torch
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt


train_data = MNIST(
    root='data',
    train=True,
    download=True,
)
test_data = MNIST(
    root='data',
    train=False,
    download=True,
)
print(train_data.data.shape, train_data.targets.shape, test_data.data.shape, test_data.targets.shape)
plt.figure(figsize=(24, 3))
plt.suptitle("MNIST", fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(train_data.data[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(str(train_data.targets.numpy()[i]), fontsize=30)
plt.show()

train_data = CIFAR10(
    root='data',
    train=True,
    download=True,
)
test_data = CIFAR10(
    root='data',
    train=False,
    download=True,
)
print(
    torch.tensor(train_data.data).shape, 
    torch.tensor(train_data.targets).shape, 
    torch.tensor(test_data.data).shape, 
    torch.tensor(test_data.targets).shape
)
class_names = [
    "airplane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
plt.figure(figsize=(24, 3))
plt.suptitle("CIFAR-10", fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(train_data.data[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_names[train_data.targets[i]], fontsize=30)
plt.show()
