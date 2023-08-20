import cv2
import torch
import numpy as np
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision import transforms
from torchvision.transforms import ToTensor


device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

transform = transforms.Compose([
    ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img = cv2.imread("rabbit.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = transform(img).to(device).unsqueeze(0)

preds = model(x)
preds = torch.softmax(preds, dim=1)
top5 = torch.topk(preds, k=5)
print("예측 결과:", top5)

with open("imagenet_classes.txt", "r") as f:
    classes = f.readlines()
classes = [c.rstrip() for c in classes]

for i in range(5):
    cv2.putText(
        img,
        classes[int(top5.indices.squeeze(0)[i])] + ":" + str(top5.values.squeeze(0)[i].item()),
        (10, 20 + i * 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Recognition result", img)

cv2.waitKey()
cv2.destroyAllWindows()
