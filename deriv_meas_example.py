import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


n_in = 1
n_h1 = 100
n_h2 = 50
n_h3 = 50
n_o = 1


class DerivTanh(torch.nn.Module):
    def __init__(self):
        super(DerivTanh, self).__init__()

    def forward(self, x):
        return 4 / (torch.exp(-x.t()) + torch.exp(x.t())).pow(2)


class DerivNet(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet, self).__init__()
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

        dh1dx = w1.repeat(1, nx)
        dh2dz1 = w2
        dydz2 = w3

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)

        dydx = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx))).t()

        # print('size: ', dydx.size())
        return (y, dydx)

model = DerivNet(n_in, n_h1, n_h2, n_o)

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100).unsqueeze(1)
(y, dydx) = model(train_x)

# True function is sin(2*pi*x) with Gaussian noise
train_y = (2 * math.pi)*torch.cos(train_x * (2 * math.pi))+torch.randn(train_x.size()) * 0.05


val_x = torch.linspace(0, 1, 100).unsqueeze(1)
# True function is sin(2*pi*x) with Gaussian noise
val_y = torch.sin(val_x * (2 * math.pi)) + torch.randn(val_x.size())*0.1 + 1.0

## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1)

train_iters = 1000
loss_save = torch.empty(train_iters, 1)
val_loss_save = torch.empty(train_iters, 1)
for epoch in range(train_iters):
    train_x = torch.linspace(0, 1, 100).unsqueeze(1)
    train_y = (2 * math.pi)*torch.cos(train_x * (2 * math.pi))+torch.randn(train_x.size()) * 0.05

    optimizer.zero_grad()
    (yhat, dyhatdx) = model(train_x)
    loss = criterion(train_y, dyhatdx)
    (yhat_val, dyvaldx) = model(val_x)
    val_loss = criterion(val_y, dyvaldx)
    print('epoch: ', epoch, ' loss: ', loss.item(), 'val loss: ', val_loss.item())
    loss.backward()
    optimizer.step()
    loss_save[epoch, 0] = loss
    val_loss_save[epoch, 0] = val_loss
    # scheduler.step(epoch)



test_x = torch.linspace(0, 1, 51).unsqueeze(1)
# test_x.grad.data.zero_() # ensure the gradient is zero
with torch.no_grad():
    (prediction, dpred) = model(test_x)
#
#
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 1, figsize=(4, 6))

    # Plot training data as black stars

    # # Plot predictive means as blue line

    ax[0].plot(test_x.numpy(), prediction.detach().numpy(), 'g')

    ax[0].set_ylim([-3, 3])
    ax[0].legend(['trained model'])

    ax[1].plot(train_x.numpy(), train_y.numpy(), 'k*')
    ax[1].plot(test_x.numpy(), dpred.detach().numpy())
    ax[1].legend(['trained derivative model','derivative measurements'])
    # ax[1].plot(loss_save.detach().log().numpy())
    # ax[1].plot(val_loss_save.log().detach().numpy(), 'r')
    # ax[1].legend(['training loss', 'val loss'])
    plt.show()
