##### 2 dims image
#--------------------------------
# read jpg image to numpy
import cv2
image = cv2.imread('./data/p1ch4/image-dog/bobby.jpg')
type(image) # numpy.ndarray
image.shape # h,w,c
image

#--------------------------------
# convert numpy to pytorch tensor, shared memory
import torch
image_t = torch.from_numpy(image)
image_t.shape # h,w,c
image_t

#--------------------------------
# pytorch tensor required the shape is c,h,w
# use transpose or permute to change the layout(just only change stride and shape, not copy the data)
image_transpose = image_t.transpose(0,2).transpose(1,2)
image_transpose.shape
image_transpose.stride()
image_transpose

image_permute = image_t.permute(2,0,1)
image_permute.shape
image_permute.stride()
image_permute

image_transpose == image_permute

#--------------------------------
# if we need to create batch tensor, we can pre-allocate the memory
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
import os
data_dir = './data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir)
    if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(filenames):
    img_arr = cv2.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1)
    img_t = img_t[:3]
    batch[i] = img_t

#--------------------------------
# normalize image data to fit the model
# 当输入数据的范围为 0～1 或−1～ 1 时，神经网络表现出最佳的训练性能，这是其构建块的定义方式所决定的。
## the first way. divide by 255
batch = batch.float()
batch /= 255.0
batch
## the second way. normalize by mean and std,means (x - mean) / std
## mean = sum(xi) / count
## std = sqrt(sum(pow(xi-mean))/count)
batch = batch.float()
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c]) # calculator all batch in each channel mean (sum all batch in each channel / count)
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std
# 在这里，我们只对单个批次的图像进行归一化，因为我们还不知道如何对整个数据集进行操
# 作。处理图像时，预先计算所有训练数据的均值和标准差，然后用这些固定的、重新计算的量进行
# 相减和相除操作是一个很好的实践。

#--------------------------------

##### 3 dims image
# NxCxDxHxW
import imageio
dir_path = "/home/aico/github/HPC/pytorch/src/1_DeepLearningWithPytorch/data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_arr.shape # deep, height, width
vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)
vol.shape
