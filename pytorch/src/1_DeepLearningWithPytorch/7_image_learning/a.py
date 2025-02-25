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
