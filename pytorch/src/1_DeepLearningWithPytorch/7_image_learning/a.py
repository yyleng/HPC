from torchvision import datasets
from torchvision import transforms
data_path = '../data-unversioned/'

pil_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

cifar10 = datasets.CIFAR10(data_path, train=True, download=True,transform=pil_transform)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,transform=pil_transform)

