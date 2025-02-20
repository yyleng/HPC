import torch
import cv2
import os

# Load image
img_dir = './data'
imgs_path = [ os.path.join(img_dir,name) for name in os.listdir(img_dir) if name.endswith('.jpg') ]
img_width = 224
img_height = 224
for i,img_path in enumerate(imgs_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(img_width,img_height))
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)
    img_t[i] = img

img_t = torch.zeros(len(imgs_path),3,img_width,img_height,dtype=torch.uint8)
img_t = img_t.float()
for i in range(len(imgs_path)):
    mean = torch.mean(img_t[i])
    print(imgs_path[i]," mean is ",mean)
    for j in range(3):
        mean = torch.mean(img_t[i,j])
        print(imgs_path[i]," channel ",j," mean is ",mean)
