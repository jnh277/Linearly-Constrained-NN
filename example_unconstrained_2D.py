import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils import data



n_in = 2
n_h1 = 100
n_h2 = 50
# n_h3 = 50  # not using this layer atm
n_o = 1

n_o_uc = 2  # there are two outputs for the unconstrained case

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)


def vector_field(x, y, c1x=0.5, c1y=0.8, c2x=0.5, c2y=0.2,l1=0.18,l2=0.18):
    f3_1 = torch.exp(-0.5*(x-c1x).pow(2)/l1**2 - 0.5*(y-c1y).pow(2)/l1**2)
    f3_2 = torch.exp(-0.5 * (c2x-x).pow(2) / l2 ** 2 - 0.5 * (c2y-y).pow(2) / l2 ** 2)
    v1 = -(y-c1y) * f3_1 / l1**2 + (c2y-y)*f3_2/l2**2
    v2 = (x-c1x) * f3_1 / l1**2 - (c2x-x)*f3_2/l2**2
    return (v1, v2)


## generate data
n_data = 2000
x_train = torch.rand(n_data, 2)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_train, x2_train)
y1_train = v1 + 0.1 * torch.randn(x1_train.size())
y2_train = v2 + 0.1 * torch.randn(x1_train.size())

x_val = torch.rand(2000,2)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_val, x2_val)
y1_val = v1 + 0.1 * torch.randn(x1_val.size())
y2_val = v2 + 0.1 * torch.randn(x1_val.size())

# now put data in a convenient dataset and data loader

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x1,x2,y1,y2):
    'Initialization'
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2

  def __len__(self):
    'Denotes the total number of samples'
    return len(self.x1)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    x1 = self.x1[index]
    x2 = self.x2[index]
    y1 = self.y1[index]
    y2 = self.y2[index]

    return x1, x2, y1, y2

training_set = Dataset(x1_train,x2_train,y1_train,y2_train)

# data loader Parameters
DL_params = {'batch_size': 500,
          'shuffle': True,
          'num_workers': 4}
training_generator = data.DataLoader(training_set, **DL_params)


train_iters = 100

## train the unconstrained model
criterion = torch.nn.MSELoss()
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 25, gamma=0.25, last_epoch=-1)


loss_save_uc = torch.empty(train_iters, 1)
val_loss_save_uc = torch.empty(train_iters, 1)

for epoch in range(train_iters):
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer_uc.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)

        (vhat) = model_uc(x_train)
        y_train = torch.cat((y1_train, y2_train), 1)
        loss = criterion(y_train, vhat)
        # loss = criterion(y1_train, vhat[:,0]) + criterion(y2_train, vhat[:,1])
        print('epoch: ', epoch, ' loss: ', loss.item())
        loss.backward()
        optimizer_uc.step()
    loss_save_uc[epoch, 0] = loss

    (vhat) = model_uc(x_val)
    y_val = torch.cat((y1_val, y2_val), 1)
    val_loss = criterion(y_val, vhat)
    val_loss_save_uc[epoch,0] = val_loss
    scheduler_uc.step(epoch)

# plot the true functions
xv, yv = torch.meshgrid([torch.arange(0.0,15.0)/15.0, torch.arange(0.0,15.0)/15.0])
# the scalar potential function

(v1,v2) = vector_field(xv, yv)



# plot the predicted function
x_pred = torch.cat((xv.reshape(15*15,1), yv.reshape(15*15,1)),1)
(vpred) = model_uc(x_pred)
v1_pred = vpred[:,0]
v2_pred = vpred[:,1]

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 1, figsize=(4, 6))
    # ax.pcolor(xv,yv,f_scalar)
    ax[0].quiver(xv, yv, v1, v2)
    ax[0].quiver(xv, yv, v1_pred.reshape(15,15).detach(), v2_pred.reshape(15,15).detach(),color='r')
    ax[0].legend(['true','predicted'])

    # ax[1].plot(loss_save.detach().log().numpy())
    # ax[1].plot(val_loss_save.detach().log().numpy())
    ax[1].plot(loss_save_uc.detach().log().numpy(), linestyle='--')
    ax[1].plot(val_loss_save_uc.detach().log().numpy(), linestyle='--')
    ax[1].set_ylabel('log loss')
    plt.show()
