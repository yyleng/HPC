import torch
import numpy as np

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u) 

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

w = torch.ones(()) # w = 1
b = torch.zeros(()) # b = 0

t_p = model(t_u, w, b)
t_p

loss = loss_fn(t_p, t_c)
loss

def dloss_dtp(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs
def dtp_dw(t_u):
    return t_u
def dtp_db():
    return 1.0

def grad_fn(t_u, t_c, t_p):
    dloss_dtp_val = dloss_dtp(t_p, t_c)
    grad_w = dloss_dtp_val * dtp_dw(t_u)
    grad_b = dloss_dtp_val * dtp_db()
    return torch.stack([grad_w.sum(), grad_b.sum()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p)

        params = params - learning_rate * grad
        print(grad)

        print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

p = training_loop(
 n_epochs = 100,
 learning_rate = 1e-2,
 params = torch.tensor([1.0, 0.0]),
 t_u = t_u,
 t_c = t_c)

p # 因为 learning_rate = 1e-2, 太大了，导致无法收敛，所以 w 和 b 会变成 nan

p = training_loop(
 n_epochs = 100,
 learning_rate = 1e-4,
 params = torch.tensor([1.0, 0.0]),
 t_u = t_u,
 t_c = t_c)

p # learning_rate = 1e-4, 收敛了

# 学习率自适应方式更好,后面有介绍

p = training_loop(
 n_epochs = 1,
 learning_rate = 1e-4,
 params = torch.tensor([1.0, 0.0]),
 t_u = t_u,
 t_c = t_c)

p # weight grad is 4517.2969, bias grad is 82.6000 说明 weight 的梯度比 bias 的梯度大很多
# 这意味着权重和 偏置存在于不同的比例空间中，在这种情况下，如果学习率足够大，能够有效更新其中一个参数，
# 那么对于另一个参数来说，学习率就会变得不稳定，而一个只适合于另一个参数的学习率也不足
# 以有意义地改变前者。当然，可以给每一参数都设置一个学习率，但这样就会增加超参数的数量，得不偿失
# 有效的解决方式是对输入进行归一化

tun = 0.1 * t_u
p = training_loop(
 n_epochs = 1,
 learning_rate = 1e-2, # 这时哪怕学习率变成1e-2，也不会导致梯度爆炸
 params = torch.tensor([1.0, 0.0]),
 t_u = tun,
 t_c = t_c)

p

p = training_loop(
 n_epochs = 5000,
 learning_rate = 1e-2, # 这时哪怕学习率变成1e-2，也不会导致梯度爆炸
 params = torch.tensor([1.0, 0.0]),
 t_u = tun,
 t_c = t_c)

p

# draw the graph
from matplotlib import pyplot as plt
t_p = model(tun, *p)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
