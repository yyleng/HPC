import torch

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)

epoch = 5000

learning_rate = 1e-2 # use 1e-4 is slowly the update grad

t_un = 0.1 * t_u # normalize input data to close -1 or 1

for i in range(epoch):
    if params.grad is not None:
        params.grad.zero_() # clean the last grad
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)
    loss.backward()
    with torch.no_grad(): # disable grad calculate when update the params
        params -= learning_rate * params.grad
    if i % 500 == 0:
        print('Epoch %d, Loss %f' % (i, float(loss)))
print(params)
