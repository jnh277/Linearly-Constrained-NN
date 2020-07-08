###############################################################################
#    Linearly Constrained Neural Networks
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    Please see  see <https://www.gnu.org/licenses/> for a copy of the license
###############################################################################

# This code is supplementary material for the submission to ICML 2020,
# This code will produce the results shown in Figure 1 based off the example given in
# Section 5.3 "Real Data"

from torch.utils import data
import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import torch.nn as nn
import torch.autograd as ag


epochs = 600
batch_size = 250
n_train = 500


# import magnetic field data
mag_data = sio.loadmat('./real_data/magnetic_field_data.mat')


pos = mag_data['pos']
mag = mag_data['mag']
pos_save = pos * 1.0
mag_save = mag * 1.0
n = len(pos[:, 0])     # length of data, using all data

# apply a random shuffling to the data
perm = torch.randperm(n)
pos = pos[perm, :]
mag = mag[perm, :]

X = torch.from_numpy(pos).float()
y = torch.from_numpy(mag).float()

# define data set class

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

# split data set into training and validation
nt = min(n_train,n)
nv = n - nt


training_set = Dataset(X[0:nt,:], y[0:nt,:])
# data loader Parameters
DL_params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 2}
training_generator = data.DataLoader(training_set, **DL_params)

X_val = X[nt:n, :]
mag_val = y[nt:n, :]

## define neural network model
n_in = 3
n_h1 = 150
n_h2 = 75
n_o = 1
n_o_uc = 3

# constrained model
class DerivNet(torch.nn.Module):
    def __init__(self, base_net):
        super(DerivNet, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        return y, dydx[:,0].unsqueeze(1), dydx[:,1].unsqueeze(1), dydx[:,2].unsqueeze(1)


model = DerivNet(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))

# unconstrained model
model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)

## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=10)


train_loss = np.empty([epochs, 1])
val_loss = np.empty([epochs, 1])


def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x_train, mag_train in training_generator:
        optimizer.zero_grad()
        (yhat, y1hat, y2hat, y3hat) = model(x_train)
        vhat = torch.cat((y1hat, y2hat, y3hat), 1)
        loss = criterion(mag_train, vhat)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss/n_batches

def eval(epoch):
    model.eval()
    (yhat, y1hat, y2hat, y3hat) = model(X_val)
    vhat = torch.cat((y1hat, y2hat, y3hat), 1)
    loss = criterion(mag_val, vhat)
    return loss


print('Training constrained NN')
for epoch in range(epochs):
    train_loss[epoch, 0] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    scheduler.step(v_loss)
    val_loss[epoch, 0] = v_loss.detach().numpy()
    print('Constrained NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


# Train a standard NN
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=10)

def train_uc(epoch):
    model_uc.train()
    total_loss = 0
    n_batches = 0
    for x_train, mag_train in training_generator:
        optimizer_uc.zero_grad()
        vhat = model_uc(x_train)
        loss = criterion(mag_train, vhat)
        loss.backward()
        optimizer_uc.step()
        total_loss += loss
        n_batches += 1
    return total_loss/n_batches

def eval_uc(epoch):
    model_uc.eval()
    with torch.no_grad():
        vhat = model_uc(X_val)
        loss = criterion(mag_val, vhat)
    return loss


train_loss_uc = np.empty([epochs, 1])
val_loss_uc = np.empty([epochs, 1])

print('Training standard NN')
for epoch in range(epochs):
    train_loss_uc[epoch, 0] = train_uc(epoch).detach().numpy()
    v_loss = eval_uc(epoch)
    scheduler_uc.step(v_loss)
    val_loss_uc[epoch, 0] = v_loss.detach().numpy()
    print('Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])


#---- see how well it did -------

X_ordered = torch.from_numpy(pos_save).float()
(p_pred, m1pred, m2pred, m3pred) = model(X_ordered)




with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].plot(np.log(train_loss))
    ax[0].plot(np.log(val_loss))
    ax[0].set_ylabel('log loss')
    ax[0].set_xlabel('epochs')
    ax[0].legend(['training','validation'])
    ax[0].set_title('Our constrained NN final validation loss={0:.2g}'.format(val_loss[-1].item()))

    ax[1].plot(np.log(train_loss_uc))
    ax[1].plot(np.log(val_loss_uc))
    ax[1].set_ylabel('log loss')
    ax[1].set_xlabel('epochs')
    ax[1].legend(['training','validation'])
    ax[1].set_title('Standard NN final validation loss={0:.2g}'.format(val_loss_uc[-1].item()))

    plt.show()

    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot3D(pos_save[:,0], pos_save[:,1], pos_save[:,2],'k',linewidth=0.5)
    # plot measurements
    skip = 2
    lengths = np.sqrt(mag_save[1:-1:skip, 0]**2+mag_save[1:-1:skip, 1]**2+mag_save[1:-1:skip, 2]**2)
    for x1, y1, z1, u1, v1, w1, l in zip(pos_save[1:-1:skip,0], pos_save[1:-1:skip,1], pos_save[1:-1:skip,2], mag_save[1:-1:skip, 0], mag_save[1:-1:skip, 1], mag_save[1:-1:skip, 2], lengths):
        ax2.quiver(x1, y1, z1, u1, v1, w1, pivot='tail', length=l * 0.1, linewidth=0.5)

    # plot predictions from constrained model
    skip = 20
    lengths = np.sqrt(m1pred[1:-1:skip].detach().numpy()**2+m2pred[1:-1:skip].detach().numpy()**2+m3pred[1:-1:skip].detach().numpy()**2)
    for x1, y1, z1, u1, v1, w1, l in zip(pos_save[1:-1:skip,0], pos_save[1:-1:skip,1], pos_save[1:-1:skip,2], m1pred[1:-1:skip].detach().numpy(), m2pred[1:-1:skip].detach().numpy(), m3pred[1:-1:skip].detach().numpy(), lengths):
        ax2.quiver(x1, y1, z1, u1, v1, w1, pivot='tail', length=l * 0.1, color='red',linewidth=0.5)
    ax2.legend(['Trajectory','measurements', 'constrained model estimates'])
    plt.show()
