import torch
# # 创建一个 5行3列的随机矩阵
# x = torch.rand(5, 3)
# print(x)
# print(x[0,:])
# print(x[:,1])

# # 创建一个5行3列的零矩阵
# y = torch.zeros(5, 3, dtype=torch.long)
# x.add_(y)
# print(x)

# # reshape
# x = torch.randn(4, 4)
# print(x)
# y = x.view(16)
# print(y)
# z = x.view(-1, 8)
# print(z)
# z1 = x.reshape(-1,8)
# print(z1)
# torch.add()

# x = torch.randn(1,18,20,20)
# y1 = x.reshape(1,3,6,20,20).permute(0,1,3,4,2) # 1,3,20,20,6
# y11 = y1.reshape(7200)
# y2 = x.reshape(1,3,6,20,20).reshape(1,3,20,20,6) # 1,3,20,20,6
# y22 = y2.reshape(7200)
# print(y11.stride())  # 查看 y1 的 stride
# print(y22.stride())  # 查看 y2 的 stride

# print(y11.equal(y22))

# # 将张量转换为连续内存布局后再比较
# y11_contiguous = y11.contiguous()
# y22_contiguous = y22.contiguous()

# print(y11_contiguous.stride())  # 查看 y11 的连续内存布局
# print(y22_contiguous.stride())  # 查看 y22 的连续内存布局

# # 比较相等性
# print(y11_contiguous.equal(y22_contiguous))

x = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]).reshape(2,3,4)
y = x.permute(1,2,0)
print(x)
print(y)
