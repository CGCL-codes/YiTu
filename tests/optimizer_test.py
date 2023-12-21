import torch.nn as nn
import torch
import sys

sys.path.append('/home/nx/ningxin/DistGNN/P3')
print(sys.path)

import P3
from P3 import optimizer
# import P3.optimizer as optimizer


class Stage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        # self.relu1 = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu1(x.clone())
        return x

class Stage2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc2(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x


model = Model()
# model.dump_patches = True
optim = optimizer.AdamWithWeightStashing(model, 2, lr=0.01)
cost = nn.CrossEntropyLoss()

label = torch.empty(5, dtype=torch.long).random_(2)
x = torch.rand(5, 3)
# 如何保证前向和反向传播过程参数版本相同？
out0 = model(x)
for _ in range(10):
    out = model(x)
    loss = cost(out0, label)
    optim.zero_grad()
    # TODO: 加载数据后版本号被修改了
    optim.load_data_old_params()
    optim.load_model_old_params()
    torch.autograd.backward(loss)

    for name, param in model.named_parameters():
        print("name: {}, param: {}, grad: {}".format(name, param, param.grad))
    optim.load_new_params()
    print("-------------------")
    for name, param in model.named_parameters():
        print("name: {}, param: {}, grad: {}".format(name, param, param.grad))
    # 每次参数更新后版本号递增
    optim.step()
    print(loss)
    out0 = out
