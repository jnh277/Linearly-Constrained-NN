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

# This code is supplementary material for the submission paper submitted to
# IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, a preprint version of
# this paper can be found at https://arxiv.org/abs/2002.01600
# This code will produce the results shown in Figure 5 based off the example given in
# Section 5.1 "Simulated Divergence-Free"


import torch
from matplotlib import pyplot as plt
from torch.utils import data
import numpy as np
import torch.nn as nn
import torch.autograd as ag

description = "Train 2D constrained and unconstrained model"

epochs = 400
n_data = 200


def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


## ------------------ set up models-------------------------- ##
# set network size
n_in = 2
n_h1 = 100
n_h2 = 50
n_o = 1

# two outputs for the unconstrained network
n_o_uc = 2

# define model class
class DivFree2D(torch.nn.Module):
    def __init__(self, base_net):
        super(DivFree2D, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        return y, dydx[:,1].unsqueeze(1), -dydx[:,0].unsqueeze(1)


model = DivFree2D(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))


model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)


# pregenerate validation data
x_val = 4.0 * torch.rand(2000, 2)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_val, x2_val)
y1_val = v1 + 0.1 * torch.randn(x1_val.size())
y2_val = v2 + 0.1 * torch.randn(x1_val.size())
y_val = torch.cat((y1_val, y2_val), 1)


n_pred = 250
# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, n_pred) * 4.0 / n_pred, torch.arange(0.0, n_pred) * 4.0 / n_pred])
(v1, v2) = vector_field(xv, yv)

xv2, yv2 = torch.meshgrid([torch.arange(0.0, 20) * 4.0 / 20, torch.arange(0.0, 20) * 4.0 / 20])
(v1_2, v2_2) = vector_field(xv2, yv2)

# generate training data
x_train = torch.empty(n_data, 2)
x_train[0:int(n_data/4), 0] = 1.0 * torch.rand(int(n_data/4))
x_train[0:int(n_data/4), 1] = 4.0 * torch.rand(int(n_data/4))

x_train[int(n_data/4):2*int(n_data/4), 0] = 3+1.0 * torch.rand(int(n_data/4))
x_train[int(n_data/4):2*int(n_data/4), 1] = 4.0 * torch.rand(int(n_data/4))

x_train[2*int(n_data/4):3*int(n_data/4), 1] = 1.0 * torch.rand(int(n_data/4))
x_train[2*int(n_data/4):3*int(n_data/4), 0] = 4.0 * torch.rand(int(n_data/4))

x_train[3*int(n_data/4):4*int(n_data/4), 1] = 3+1.0 * torch.rand(int(n_data/4))
x_train[3*int(n_data/4):4*int(n_data/4), 0] = 4.0 * torch.rand(int(n_data/4))

x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(v1_t, v2_t) = vector_field(x1_train, x2_train)
y1_train = v1_t + 0.1 * torch.randn(x1_train.size())
y2_train = v2_t + 0.1 * torch.randn(x1_train.size())

# define Dataset class
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x1, x2, y1, y2):
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


training_set = Dataset(x1_train, x2_train, y1_train, y2_train)

# data loader Parameters
DL_params = {'batch_size': 100,
             'shuffle': True,
             'num_workers': 2,
             'pin_memory': False}
training_generator = data.DataLoader(training_set, **DL_params)


# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=15)


def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        (yhat, v1hat, v2hat) = model(x_train)
        loss = (criterion(y1_train, v1hat) + criterion(y2_train, v2hat)) / 2  # divide by 2 as it is a mean
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    (yhat, v1hat, v2hat) = model(x_val)
    loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat)) / 2
    return loss.cpu()


train_loss = np.empty([epochs, 1])
val_loss = np.empty([epochs, 1])

print('Training Constrained NN')

for epoch in range(epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    scheduler.step(v_loss)
    val_loss[epoch] = v_loss.detach().numpy()
    print('Constrained NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


# work out the rms error for this one
x_pred = torch.cat((xv.reshape(n_pred * n_pred, 1), yv.reshape(n_pred * n_pred, 1)), 1)
(f_pred, v1_pred, v2_pred) = model(x_pred)
error_new = torch.cat((v1.reshape(n_pred*n_pred, 1) - v1_pred.detach(), v2.reshape(n_pred*n_pred, 1) - v2_pred.detach()), 0)
rms_error = torch.sqrt(sum(error_new * error_new) / (2*n_pred*n_pred))

# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=15)
def train_uc(epoch):
    model_uc.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer_uc.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        vhat = model_uc(x_train)
        y_train = torch.cat((y1_train, y2_train), 1)
        loss = criterion(y_train, vhat)
        loss.backward()
        optimizer_uc.step()
        total_loss += loss.cpu()
        n_batches += 1
    return total_loss / n_batches

def eval_uc(epoch):
    model_uc.eval()
    with torch.no_grad():
        (vhat) = model_uc(x_val)
        loss = criterion(y_val, vhat)
    return loss.cpu()


train_loss_uc = np.empty([epochs, 1])
val_loss_uc = np.empty([epochs, 1])

print('Training standard NN')

for epoch in range(epochs):
    train_loss_uc[epoch] = train_uc(epoch).detach().numpy()
    v_loss = eval_uc(epoch)
    scheduler_uc.step(v_loss)
    val_loss_uc[epoch] = v_loss.detach().numpy()
    print('Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])

# work out final rms error for unconstrainted net
# work out the rms error for this trial
(v_pred_uc) = model_uc(x_pred)
v1_pred_uc = v_pred_uc[:, 0]
v2_pred_uc = v_pred_uc[:, 1]

error_uc = torch.cat((v1.reshape(n_pred*n_pred) - v1_pred_uc.detach(), v2.reshape(n_pred*n_pred) - v2_pred_uc.detach()), 0)
rms_uc = torch.sqrt(sum(error_uc * error_uc) / (2*n_pred*n_pred))

# ----------------- save configuration options and results -------------------------------


## determine constraint violations
with torch.no_grad():
    v1_pred_mat = v1_pred.reshape(n_pred,n_pred)
    v2_pred_mat = v2_pred.reshape(n_pred,n_pred)


    dx = 4.0/n_pred
    dy = 4.0/n_pred
    dfdy = torch.empty(n_pred,n_pred)
    dfdy[:,0] = (v2_pred_mat[:,1] - v2_pred_mat[:,0])/dy
    dfdy[:,-1] = (v2_pred_mat[:,-1] - v2_pred_mat[:,-2])/dy
    dfdy[:,1:-1] = (v2_pred_mat[:,1:-1] - v2_pred_mat[:,0:-2])/dy/2 + (v2_pred_mat[:,2:] - v2_pred_mat[:,1:-1])/dy/2

    dfdx = torch.empty(n_pred,n_pred)
    dfdx[0,:] = (v1_pred_mat[1,:] - v1_pred_mat[0,:])/dx
    dfdx[-1,:] = (v1_pred_mat[-1,:] - v1_pred_mat[-2,:])/dx
    dfdx[1:-1,:] = (v1_pred_mat[1:-1,:] - v1_pred_mat[0:-2,:])/dx/2 + (v1_pred_mat[2:,:] - v1_pred_mat[1:-1, :])/dx/2

    Cviol = dfdx + dfdy

    v1_pred_mat = v1_pred_uc.reshape(n_pred,n_pred)
    v2_pred_mat = v2_pred_uc.reshape(n_pred,n_pred)
    dfdy = torch.empty(n_pred,n_pred)
    dfdy[:,0] = (v2_pred_mat[:,1] - v2_pred_mat[:,0])/dy
    dfdy[:,-1] = (v2_pred_mat[:,-1] - v2_pred_mat[:,-2])/dy
    dfdy[:,1:-1] = (v2_pred_mat[:,1:-1] - v2_pred_mat[:,0:-2])/dy/2 + (v2_pred_mat[:,2:] - v2_pred_mat[:,1:-1])/dy/2

    dfdx = torch.empty(n_pred,n_pred)
    dfdx[0,:] = (v1_pred_mat[1,:] - v1_pred_mat[0,:])/dx
    dfdx[-1,:] = (v1_pred_mat[-1,:] - v1_pred_mat[-2,:])/dx
    dfdx[1:-1,:] = (v1_pred_mat[1:-1,:] - v1_pred_mat[0:-2,:])/dx/2 + (v1_pred_mat[2:,:] - v1_pred_mat[1:-1, :])/dx/2

    Cviol_uc = dfdx + dfdy


with torch.no_grad():
    # Initialize second plot
    f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
    Q = ax2[0].quiver(xv2, yv2, v1_2, v2_2, scale=None, scale_units='inches')
    Q._init()
    assert isinstance(Q.scale, float)
    ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    ax2[0].plot([1.0, 3.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 3.0, 1.0],'--')
    ax2[0].set_xlabel('$x_1$')
    ax2[0].set_ylabel('$x_2$')

    ax2[1].pcolor(xv, yv, Cviol, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    ax2[1].set_xlabel('$x_1$')
    ax2[1].set_ylabel('$x_2$')
    ax2[1].set_title('Our Constrained Approach')


    ax2[2].pcolor(xv, yv, Cviol_uc, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    ax2[2].set_xlabel('$x_1$')
    ax2[2].set_ylabel('$x_2$')
    ax2[2].set_title('Standard NN')
    plt.show()



