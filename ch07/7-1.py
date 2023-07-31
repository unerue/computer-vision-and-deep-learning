import tensorflow as tf
import matplotlib.pyplot as plt
from torchvision import datasets


datasets.MNIST()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

plt.figure(figsize=(24, 3))
plt.suptitle("MNIST", fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.title(str(y_train[i]), fontsize=30)
plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
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

trainset = datasets.MNIST(train=False, download=True)
plt.figure(figsize=(24, 3))
plt.suptitle("CIFAR-10", fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_names[y_train[i, 0]], fontsize=30)
plt.show()
