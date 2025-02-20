import torch

## 1. add name to each dims (avoid many mistakes in the future)
t1 = torch.randn(3,5,5,names=['channel','height','width'])
t3 = torch.randn(4,3,5,5)
# add names to t3
t3.refine_names(...,'channel','height','width') # ... means keep ignore the first dim
# change or delete name
t1.rename(None,'h','w') # delete name
t1.rename(None) # delete name

## 2. expand dims
t2 = torch.ones(3)
t2.shape # 3
t2.unsqueeze(0).shape # 1,3
t2.unsqueeze(-1).shape # 3,1
t2[None].shape # 1,3
t2[:,None].shape # 3,1

## 3. einsum
img = torch.randn(3,5,5)
weights = torch.randn(3)
img_weights = (img * weights[:,None,None]).sum(-3)
print("img_weights.shape:",img_weights.shape) # 5,5
# equal to the above line
img_weights_fancy = torch.einsum('...chw,c->...hw',img,weights)
img_weights_fancy == img_weights

## 4. dtype
t4 = torch.randn(3,5,5,dtype=torch.float16)
t4.to(torch.float32).dtype # to will check whether convert
t4.float().dtype
float(t4[0,0,0]) # float() support one element tensor

## 5. storage
# whatever the tensor is, the storage is a 1D array
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage() # [torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 6]
points_storage[0] # manual access storage
points_storage[1] = 2.0 # manual change storage value

# change value in tensor in placed
a = torch.ones(3, 2)
a.zero_() # *_() function will change the tensor value in storage
a

## 6. size,stride,offset
points = torch.randn(3,4,5)
points.shape
points.stride() # (20, 5, 1)
points[1].storage_offset()
points[1,1].storage_offset()
points[1,1,1].storage_offset()

## 7. clone 
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone() # clone will copy the value in storage but not the storage itself
second_point[0] = 10.0
points

## 8. transpose in placed
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.stride()
points_t = points.t() # t() is the specific function for 2D tensor transpose
points_t.stride()
id(points.storage()) == id(points_t.storage()) # True, store in same storage

points = torch.randn(3,4,5)
points.stride()
points_t = points.transpose(0,1)
points_t.stride()
id(points.storage()) == id(points_t.storage()) # True, store in same storage
# transpose means corresponding stride and shape change, but not change the storage

### 9. contiguous
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.stride() # 2,1
points.shape # 3,2
points.storage() # 4,1,5,3,2,1
points_t = points.t()
points_t.stride() # 1,2
points_t.shape # 2,3
points.is_contiguous() # True
points_t.is_contiguous() # False
points_t.storage() # 4,1,5,3,2,1

points_cont = points_t.contiguous()
points_cont.stride() # 3,1
points_cont.shape # 2,3
points_cont.storage() # 4,5,2,1,3,1

id(points.storage()) == id(points_t.storage()) # True, store in same storage
id(points.storage()) == id(points_cont.storage()) # True, store in same storage

### 10. cuda gpu
import time
s = time.time()
points_cpu = torch.randn(100,1000,1000,dtype=torch.float32)
result = points_cpu * points_cpu
e = time.time()
e-s # 0.4500565528869629

s = time.time()
points_gpu = torch.randn(100,1000,1000,dtype=torch.float32,device='cuda:0') # or .to(device='cuda') # or .cuda() # or .cuda(1)
result = points_gpu * points_gpu
e = time.time()
e-s # 0.009733915328979492
result.cpu() # or .to(device='cpu')

### 11. numpy
points = torch.ones(3, 4)
points_np = points.numpy() # if points is on gpu, it will raise error
points_np
points = torch.from_numpy(points_np)
points

### 12. save and load tensor
points = torch.ones(3, 4)
torch.save(points,'points.t')
points_load = torch.load('points.t',weights_only=True)

# use h5py to save and load tensor(general HDF5 format)
import h5py
points = torch.ones(3, 4)
f = h5py.File('ourpoints.hdf5', 'w')
dset = f.create_dataset('coords',data=points.numpy()) # coords is the key, you can set any value
f.close()

f = h5py.File('ourpoints.hdf5', 'r')
dset = f['coords']
pt = torch.from_numpy(dset[:]) # copy from numpy array, not share the storage
pt
