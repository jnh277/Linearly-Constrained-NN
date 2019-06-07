import scipy.io as sio
from torch.utils import data
import math
import torch
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mag_data=sio.loadmat('/Users/johannes/Documents/GitHub/Linearly-Constrained-NN/real_data/magnetic_field_data.mat')

pos = mag_data['pos']
mag = mag_data['mag']

n = len(pos[:, 0])     # length of data

# apply a random shuffling to the data
perm = torch.randperm(n)
pos = pos[perm, :]
mag = mag[perm, :]

# normalising inputs (would this effect the constraints)???
min_x = pos[:, 0].min()
min_y = pos[:, 1].min()
min_z = pos[:, 2].min()

max_x = pos[:, 0].max()
max_y = pos[:, 1].max()
max_z = pos[:, 2].max()

# X = pos.copy()
X = torch.from_numpy(pos).float()
X[:, 0] = (X[:, 0]-min_x)/(max_x-min_x)*2.0 - 1.0
X[:, 1] = (X[:, 1]-min_y)/(max_y-min_y)*2.0 - 1.0
X[:, 2] = (X[:, 2]-min_z)/(max_z-min_z)*2.0 - 1.0

y = torch.from_numpy(mag).float()



class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, mag):
    'Initialization'
    self.X = X
    self.mag = mag


  def __len__(self):
    'Denotes the total number of samples'
    return len(self.X[:,0])

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    X = self.X[index,:]
    mag = self.mag[index,:]

    return X, mag

nv = math.floor(n*0.1)
nt = n - nv

training_set = Dataset(X[0:nt,:], y[0:nt,:])
# data loader Parameters
DL_params = {'batch_size': 500,
          'shuffle': True,
          'num_workers': 4}
training_generator = data.DataLoader(training_set, **DL_params)

X_val = X[nt:n, :]
mag_val = y[nt:n, :]

## define neural network model
n_in = 3
n_h1 = 100
n_h2 = 50
n_h3 = 25
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

## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

train_iters = 40
loss_save = torch.empty(train_iters, 1)
val_loss_save = torch.empty(train_iters, 1)


for epoch in range(train_iters):
    for x_train, mag_train in training_generator:
        optimizer.zero_grad()
        (yhat, y1hat, y2hat, y3hat) = model(x_train)
        loss = criterion(mag_train[:,0], y1hat) + criterion(mag_train[:,1], y2hat) + \
               criterion(mag_train[:,2], y3hat)
        print('epoch: ', epoch, ' loss: ', loss.item())
        loss.backward()
        optimizer.step()
    loss_save[epoch, 0] = loss

    (yhat, y1hat, y2hat, y3hat) = model(x_train)
    val_loss = criterion(mag_val[:,0], y1hat) + criterion(mag_val[:,1], y2hat) + \
               criterion(mag_val[:,2], y3hat)
    val_loss_save[epoch,0] = val_loss
    scheduler.step(epoch)


# generate quiver plot data
grid_x, grid_y= np.meshgrid(np.arange(-1.0, 1.0, 0.2),
                      np.arange(-1.0, 1.0, 0.2))
grid_z = 0.35*np.ones(np.shape(grid_x))

mag_x_interp = griddata(X.numpy(), mag[:,0], (grid_x, grid_y, grid_z), method='linear')
mag_y_interp = griddata(X.numpy(), mag[:,1], (grid_x, grid_y, grid_z), method='linear')

xv = torch.from_numpy(grid_x).float()
yv = torch.from_numpy(grid_y).float()
zv = torch.from_numpy(grid_z).float()

X_pred = torch.cat((xv.reshape(10*10,1), yv.reshape(10*10,1), zv.reshape(10*10,1)),1)
(fpred, f1pred, f2pred, f3pred) = model(X_pred)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 1, figsize=(4, 6))
    # ax.pcolor(xv,yv,f_scalar)
    ax[0].quiver(grid_x, grid_y, mag_x_interp, mag_y_interp)
    ax[0].quiver(grid_x, grid_y, f1pred.reshape(10,10).detach(), f2pred.reshape(10,10).detach(),color='r')
    # ax[0].legend(['true','predicted'])

    ax[1].plot(loss_save.detach().log().numpy())
    ax[1].plot(val_loss_save.detach().log().numpy(),color='r')
    ax[1].set_ylabel('log loss')
    ax[1].set_xlabel('epochs')
    plt.show()
