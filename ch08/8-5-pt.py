import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


train_data = CIFAR10(
    root="data",
    train=True,
    download=True,
)
test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
)
x_train = train_data.data[:15]
y_train = train_data.targets[:15]  # 앞 15개에 대해서만 증대 적용
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "flog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(20, 2))
plt.suptitle("First 15 images in the train set")
for i in range(15):
    plt.subplot(1, 15, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_names[y_train[i]])
plt.show()

batch_size = 4  # 한 번에 생성하는 양(미니 배치)
transform = transforms.Compose([
    ToTensor(),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), fill=None),
    transforms.RandomHorizontalFlip(),
])

for a in range(3):
    img = x_train[a*batch_size:(a+1)*batch_size]
    label = y_train[a*batch_size:(a+1)*batch_size]
    plt.figure(figsize=(8, 2.4))
    plt.suptitle("Generatior trial " + str(a + 1))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(transform(img[i]).numpy().transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.title(class_names[label[i]])
    plt.show()
