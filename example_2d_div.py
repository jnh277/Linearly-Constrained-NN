import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


n_in = 2
n_h1 = 100
n_h2 = 50
n_h3 = 50
n_o = 1

class DerivTanh(torch.nn.Module):
    def __init__(self):
        super(DerivTanh, self).__init__()

    def forward(self, x):
        return 4 / (torch.exp(-x.t()) + torch.exp(x.t())).pow(2)


class DerivNet2D(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet2D, self).__init__()
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

        # derivative of h2 with respect to x2 (y-direction)
        dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)

        dh2dz1 = w2
        dydz2 = w3

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)

        # derivative of output with respect to x1
        dydx1 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()

        # derivative of output with respect to x2
        dydx2 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()

        # print('size: ', dydx.size())
        v1 = dydx2
        v2 = -dydx1
        return (y, v1, v2)

model = DerivNet2D(n_in, n_h1, n_h2, n_o)


def vector_field(x, y, c1x=0.5, c1y=0.8, c2x=0.5, c2y=0.2,l1=0.18,l2=0.18):
    f3_1 = torch.exp(-0.5*(x-c1x).pow(2)/l1**2 - 0.5*(y-c1y).pow(2)/l1**2)
    f3_2 = torch.exp(-0.5 * (c2x-x).pow(2) / l2 ** 2 - 0.5 * (c2y-y).pow(2) / l2 ** 2)
    v1 = -(y-c1y) * f3_1 / l1**2 + (c2y-y)*f3_2/l2**2
    v2 = (x-c1x) * f3_1 / l1**2 - (c2x-x)*f3_2/l2**2
    return (v1, v2)


x_train = torch.rand(500,2)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_train, x2_train)
y1_train = v1 + 0.1 * torch.randn(x1_train.size())
y2_train = v2 + 0.1 * torch.randn(x1_train.size())




## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 750, gamma=0.25, last_epoch=-1)

train_iters = 2000
loss_save = torch.empty(train_iters, 1)
# val_loss_save = torch.empty(train_iters, 1)

for epoch in range(train_iters):
    x_train = torch.rand(1000, 2)
    x1_train = x_train[:, 0].unsqueeze(1)
    x2_train = x_train[:, 1].unsqueeze(1)

    (v1, v2) = vector_field(x1_train, x2_train)
    y1_train = v1 + 0.1 * torch.randn(x1_train.size())
    y2_train = v2 + 0.1 * torch.randn(x1_train.size())

    optimizer.zero_grad()
    (yhat, v1hat, v2hat) = model(x_train)
    loss = criterion(y1_train, v1hat) + criterion(y2_train, v2hat)
    print('epoch: ', epoch, ' loss: ', loss.item())
    # (yhat_val, dyvaldx) = model(val_x)
    # val_loss = criterion(val_y, dyvaldx)
    # print('epoch: ', epoch, ' loss: ', loss.item(), 'val loss: ', val_loss.item())
    loss.backward()
    optimizer.step()
    loss_save[epoch, 0] = loss
    # val_loss_save[epoch, 0] = val_loss
    scheduler.step(epoch)

# plot the true functions
xv, yv = torch.meshgrid([torch.arange(0.0,15.0)/15.0, torch.arange(0.0,15.0)/15.0])
# the scalar potential function

(v1,v2) = vector_field(xv, yv)



# plot the predicted function
x_pred = torch.cat((xv.reshape(15*15,1), yv.reshape(15*15,1)),1)
(f_pred, v1_pred, v2_pred) = model(x_pred)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 1, figsize=(4, 6))
    # ax.pcolor(xv,yv,f_scalar)
    ax[0].quiver(xv, yv, v1, v2)
    ax[0].quiver(xv, yv, v1_pred.reshape(15,15).detach(), v2_pred.reshape(15,15).detach(),color='r')
    ax[0].legend(['true','predicted'])

    ax[1].plot(loss_save.detach().log().numpy())
    ax[1].set_ylabel('log loss')
    plt.show()
