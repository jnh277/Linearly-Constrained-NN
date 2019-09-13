import scipy.io as sio
from torch.utils import data
import math
import torch
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import models
# from mpl_toolkits.mplot3d import Axes3D

mag_data=sio.loadmat('/Users/johannes/Documents/GitHub/Linearly-Constrained-NN/real_data/magnetic_field_data.mat')

pos = mag_data['pos']
mag = mag_data['mag']

pos_save = pos.copy()
mag_save = mag.copy()

n = len(pos[:, 0])     # length of data, using all data

# apply a random shuffling to the data
torch.manual_seed(2)
perm = torch.randperm(n)
pos = pos[perm, :]
mag = mag[perm, :]

# normalising inputs (would this effect the constraints)??? maybe
min_x = pos[:, 0].min()
min_y = pos[:, 1].min()
min_z = pos[:, 2].min()

max_x = pos[:, 0].max()
max_y = pos[:, 1].max()
max_z = pos[:, 2].max()

# X = pos.copy()
X = torch.from_numpy(pos).float()
# X[:, 0] = (X[:, 0]-min_x)/(max_x-min_x)*2.0 - 1.0
# X[:, 1] = (X[:, 1]-min_y)/(max_y-min_y)*2.0 - 1.0
# X[:, 2] = (X[:, 2]-min_z)/(max_z-min_z)*2.0 - 1.0

min_mag1 = mag[:, 0].min()
min_mag2 = mag[:, 1].min()
min_mag3 = mag[:, 2].min()

max_mag1 = mag[:, 0].max()
max_mag2 = mag[:, 1].max()
max_mag3 = mag[:, 2].max()

y = torch.from_numpy(mag).float()

# # see if output scaling helps (didnt help)
# y[:, 0] = (y[:, 0] - min_mag1)/(max_mag1-min_mag1)*2.0 - 1.0
# y[:, 1] = (y[:, 1] - min_mag2)/(max_mag2-min_mag2)*2.0 - 1.0
# y[:, 2] = (y[:, 2] - min_mag3)/(max_mag3-min_mag3)*2.0 - 1.0



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

nv = math.floor(n*0.3)
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
# n_h3 = 2
n_o = 1



model = models.DerivNet3D(n_in, n_h1, n_h2, n_h3, n_o)

## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)

train_iters = 350
loss_save = torch.empty(train_iters, 1)
val_loss_save = torch.empty(train_iters, 1)


for epoch in range(train_iters):
    for x_train, mag_train in training_generator:
        optimizer.zero_grad()
        (yhat, y1hat, y2hat, y3hat) = model(x_train)
        vhat = torch.cat((y1hat,y2hat,y3hat), 1)
        # loss = (criterion(mag_train[:,0], y1hat) + criterion(mag_train[:,1], y2hat) +
        #         criterion(mag_train[:,2], y3hat))/3 # /3 for mean
        loss = criterion(mag_train, vhat)

        loss.backward()
        optimizer.step()
    loss_save[epoch, 0] = loss

    (yhat, y1hat, y2hat, y3hat) = model(X_val)
    vhat = torch.cat((y1hat, y2hat, y3hat), 1)
    # val_loss = (criterion(mag_val[:,0], y1hat) + criterion(mag_val[:,1], y2hat) +
    #             criterion(mag_val[:,2], y3hat))/3 # /3 for mean
    val_loss = criterion(mag_val, vhat)
    print('epoch: ', epoch, 'val loss: ', val_loss.item())
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

    # fig2 = plt.figure()
    # ax2 = plt.axes(projection='3d')
    # ax2.plot3D(pos_save[:,0], pos_save[:,1], pos[:,2], color='blue')
    plt.show()
