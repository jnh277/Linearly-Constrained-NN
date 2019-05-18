import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


n_in = 3
n_h1 = 200
n_h2 = 100
n_h3 = 50
n_o = 1

class DerivTanh(torch.nn.Module):
    def __init__(self):
        super(DerivTanh, self).__init__()

    def forward(self, x):
        return 4 / (torch.exp(-x.t()) + torch.exp(x.t())).pow(2)


class DerivNet3D(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet3D, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_h1)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_h1, n_h2)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(n_h2, n_o)
        self.derivTanh1 = DerivTanh()
        self.derivTanh2 = DerivTanh()

    def forward(self, x):
        h1 = self.linear1(x)
        z1 = self.tanh1(h1)
        h2 = self.linear2(z1)
        z2 = self.tanh2(h2)
        y = self.linear3(z2)

        # differential model
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        w3 = self.linear3.weight

        # derivative of h1 with respect to x1 (x-drection)
        dh1dx1 = w1[:,0].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x2 (y-direction)
        dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x3 (z-direction)
        dh1dx3 = w1[:, 2].unsqueeze(1).repeat(1, nx)

        dh2dz1 = w2
        dydz2 = w3

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)

        # derivative of output with respect to x1
        dydx1 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()

        # derivative of output with respect to x2
        dydx2 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()

        # derivative of output with respect to x3
        dydx3 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx3))).t()

        # print('size: ', dydx.size())
        return (y, dydx1, dydx2, dydx3)

model = DerivNet3D(n_in, n_h1, n_h2, n_o)

x_train = torch.rand(500, 3)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)
x3_train = x_train[:, 2].unsqueeze(1)
y1_train = (math.pi) * torch.cos(1 * math.pi * x1_train) * torch.cos(2 * math.pi * x2_train) * x3_train.pow(2)
y2_train = -(2 * math.pi) * torch.sin(1 * math.pi * x1_train) * torch.sin(2 * math.pi * x2_train) * x3_train.pow(2)
y3_train = 2*torch.sin(1*math.pi*x1_train) * torch.cos(2*math.pi * x2_train) * x3_train


## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 750, gamma=0.25, last_epoch=-1)

train_iters = 5000
loss_save = torch.empty(train_iters, 1)

for epoch in range(train_iters):
    x_train = torch.rand(1000, 3)
    x1_train = x_train[:, 0].unsqueeze(1)
    x2_train = x_train[:, 1].unsqueeze(1)
    x3_train = x_train[:, 2].unsqueeze(1)
    y1_train = (math.pi) * torch.cos(1 * math.pi * x1_train) * torch.cos(2 * math.pi * x2_train) * x3_train.pow(2)
    y2_train = -(2 * math.pi) * torch.sin(1 * math.pi * x1_train) * torch.sin(2 * math.pi * x2_train) * x3_train.pow(2)
    y3_train = 2 * torch.sin(1 * math.pi * x1_train) * torch.cos(2 * math.pi * x2_train) * x3_train

    optimizer.zero_grad()
    (yhat, dyhatdx1,dyhatdx2,dyhatdx3) = model(x_train)
    loss = criterion(y1_train, dyhatdx1) + criterion(y2_train, dyhatdx2) + criterion(y3_train, dyhatdx3)
    print('epoch: ', epoch, ' loss: ', loss.item())
    # (yhat_val, dyvaldx) = model(val_x)
    # val_loss = criterion(val_y, dyvaldx)
    # print('epoch: ', epoch, ' loss: ', loss.item(), 'val loss: ', val_loss.item())
    loss.backward()
    optimizer.step()
    loss_save[epoch, 0] = loss
    # val_loss_save[epoch, 0] = val_loss
    scheduler.step(epoch)