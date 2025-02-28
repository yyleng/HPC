from torchvision import datasets
from torchvision import transforms
data_path = '../data-unversioned/'

pil_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

cifar10 = datasets.CIFAR10(data_path, train=True, download=True,transform=pil_transform)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,transform=pil_transform)

# 飞机（0）
# 汽车（1）
# 鸟（2）
# 猫（3）
# 鹿（4）
# 狗（5）
# 青蛙（6）
# 马（7）
# 船 （8）
# 卡车（9）

# only keep airplane and bird
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

import torch.nn as nn
import torch.optim as optim
import torch
dataloader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=True)

in_feature = 32 * 32 * 3
out_feature = 2
model = nn.Sequential(
    nn.Linear(in_feature, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, out_feature),
    nn.LogSoftmax(dim=1)
)

loss_fn = nn.NLLLoss()
epochs = 100
optimizer = optim.SGD(model.parameters(), lr=1e-2)

for i in range(epochs):
    for imgs, labels in dataloader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {i}, Loss: {float(loss)}')

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in dataloader_val:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print(f'val Accuracy: {correct / total}')

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in dataloader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print(f'train Accuracy: {correct / total}')
