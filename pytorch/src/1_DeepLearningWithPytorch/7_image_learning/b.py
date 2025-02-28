import torch
import torch.nn as nn

# 定义 LogSoftmax 层
log_softmax = nn.LogSoftmax(dim=1)

# 输入 logits
logits = torch.tensor([[2.0, 1.0, 1.0]])

# 计算 LogSoftmax
log_probs = log_softmax(logits)

print('LogSoftmax Output:', log_probs)

loss = nn.NLLLoss()
target = torch.tensor([1])
l = loss(log_probs, target)
print('NLLLoss:', l)
