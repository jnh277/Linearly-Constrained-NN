###############################################################################
#    Linearly Constrained Neural Networks
#    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.au >
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
# This code will run the example from Section 5.2 "Simulated Strain Field" and produce
# figure 6
# The code uses our proposed approach to train a constrained neural network as well as training
# a standard neural network on simulated measurements of a strain field. Both networks are then
# used to give predictions of the strain field.

import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils import data
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.autograd as ag

torch.manual_seed(4)
epochs = 600
n_data = 200
batch_size = 100


# Define the strain field
l = 20e-3
h = 10e-3
def strain_field(x,y, P=2e3, E=200e9,l=20e-3,h=10e-3,t=5e-3,nu=0.3):
    I = t*h*h*h/12
    Exx = P/E/I*(l-x)*y
    Eyy = -nu*P/E/I*(l-x)*y
    Exy = -(1+nu)*P/2/E/I*((h/2)*(h/2) - y*y)
    return Exx, Eyy, Exy


# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, 100.0) * l / 100.0, torch.arange(0.0, 50.0) * h / 50.0-h/2])
(Exx_gv, Eyy_gv, Exy_gv) = strain_field(xv, yv, l=l,h=h)


# set network size
n_in = 2
n_h1 = 20
n_h2 = 10
n_h3 = 5
n_o = 1

# three outputs for the unconstrained network
n_o_uc = 3

# define strain field model class
class Strain2d(torch.nn.Module):
    def __init__(self, base_net, nu=0.28):
        super(Strain2d, self).__init__()
        self.base_net = base_net
        self.nu = nu

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        g = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        hx = ag.grad(outputs=g[:, 0], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:, 0].size()),
                       retain_graph=True, only_inputs=True)[0]
        hy = ag.grad(outputs=g[:, 1], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:, 1].size()),
                     retain_graph=True, only_inputs=True)[0]
        Exx = hy[:, 1].unsqueeze(1) - self.nu * hx[:, 0].unsqueeze(1)
        Eyy = hx[:, 0].unsqueeze(1) - self.nu * hy[:, 1].unsqueeze(1)
        Exy = - (1 + self.nu) * hx[:, 1].unsqueeze(1)
        return Exx, Eyy, Exy, y

model = Strain2d(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_h3),nn.Tanh(),
                                         nn.Linear(n_h3,n_o)),nu=0.3)

# create a standard neural network

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_h3),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h3, n_o_uc),
)

# define measurement noise
sigma = 2.5e-4

# pregenerate validation data
x_val = torch.cat((l*torch.rand(2000, 1),-h/2+h*torch.rand(2000, 1)),1)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(Exx, Eyy, Exy) = strain_field(x1_val, x2_val, l=l, h=h)
Exx_val = Exx + sigma * torch.randn(x1_val.size())
Eyy_val = Eyy + sigma * torch.randn(x1_val.size())
Exy_val = Exy + sigma * torch.randn(x1_val.size())
E_val = torch.cat((Exx_val, Eyy_val, Exy_val), 1)

# generate training data
x_train = torch.cat((l*torch.rand(n_data, 1),-h/2+h*torch.rand(n_data, 1)),1)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(Exx, Eyy, Exy) = strain_field(x1_train, x2_train, l=l, h=h)
Exx_train = Exx + sigma * torch.randn(x1_train.size())
Eyy_train = Eyy + sigma * torch.randn(x1_train.size())
Exy_train = Exy + sigma * torch.randn(x1_train.size())


# define some data scaling parameters
sc = 2e2
p_scale = 0.1

# define a dataset class
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x1, x2, Exx, Eyy, Exy):
        'Initialization'
        self.x1 = x1
        self.x2 = x2
        self.Exx = Exx
        self.Eyy = Eyy
        self.Exy = Exy

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x1 = self.x1[index]
        x2 = self.x2[index]
        Exx = self.Exx[index]
        Eyy = self.Eyy[index]
        Exy = self.Exy[index]

        return x1, x2, Exx, Eyy, Exy

# create the data set and data loader

training_set = Dataset(x1_train, x2_train, Exx_train, Eyy_train, Exy_train)

# data loader Parameters
DL_params = {'batch_size': batch_size,
             'shuffle': True,
             'num_workers': 2,
             'pin_memory': False}
training_generator = data.DataLoader(training_set, **DL_params)



# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=75,
                                                     min_lr=1e-10,
                                                     factor=0.25,
                                                    cooldown=50)

def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, Exx_train, Eyy_train, Exy_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        (Exx, Eyy, Exy, y) = model(x_train*sc)
        loss = (criterion(Exx_train/p_scale, Exx)
                + criterion(Eyy_train/p_scale, Eyy)
                + criterion(Exy_train/p_scale, Exy)) / 3  # divide by 2 as it is a mean
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    # with torch.no_grad():
    (Exx, Eyy, Exy, y) = model(x_val*sc)
    loss = (criterion(Exx_val/p_scale, Exx)
            + criterion(Eyy_val/p_scale, Eyy)
            + criterion(Exy_val/p_scale, Exy)) / 3
    return loss


train_loss = np.empty([epochs, 1])
val_loss = np.empty([epochs, 1])

print('Training constrained NN')

for epoch in range(epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    scheduler.step(v_loss)
    val_loss[epoch] = v_loss.detach().numpy()
    print('Constrained NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])

# determine rms error and plots
x_pred = torch.cat((xv.reshape(-1,1), yv.reshape(-1,1)), 1)
(Exx_p, Eyy_p, Exy_p, y) = model(x_pred*sc)
Exx_p = Exx_p*p_scale
Eyy_p = Eyy_p*p_scale
Exy_p = Exy_p*p_scale
error_new = torch.cat((Exx_gv.view(-1,1) - Exx_p.detach(), Eyy_gv.reshape(-1, 1) - Eyy_p.detach(),
                       Exy_gv.reshape(-1, 1) - Exy_p.detach()), 0)
rms = torch.sqrt((error_new * error_new).mean())
mae = error_new.abs().mean()


# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=25)

def train_uc(epoch):
    model_uc.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, Exx_train, Eyy_train, Exy_train in training_generator:
        optimizer_uc.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        Ehat = model_uc(x_train*sc)
        Exx = Ehat[:, 0].unsqueeze(1)
        Eyy = Ehat[:, 1].unsqueeze(1)
        Exy = Ehat[:, 2].unsqueeze(1)
        loss = (criterion(Exx_train / p_scale, Exx)
                + criterion(Eyy_train / p_scale, Eyy)
                + criterion(Exy_train / p_scale, Exy)) / 3  # divide by 2 as it is a mean
        loss.backward()
        optimizer_uc.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval_uc(epoch):
    model_uc.eval()
    with torch.no_grad():
        Ehat = model_uc(x_val*sc)
        Exx = Ehat[:, 0].unsqueeze(1)
        Eyy = Ehat[:, 1].unsqueeze(1)
        Exy = Ehat[:, 2].unsqueeze(1)
        loss = (criterion(Exx_val / p_scale, Exx)
                + criterion(Eyy_val / p_scale, Eyy)
                + criterion(Exy_val / p_scale, Exy)) / 3
    return loss


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
(Ehat_uc) = model_uc(x_pred*sc)
Exx_uc = Ehat_uc[:, 0]*p_scale
Eyy_uc = Ehat_uc[:, 1]*p_scale
Exy_uc = Ehat_uc[:, 2]*p_scale

error_new = torch.cat((Exx_gv.view(-1,1) - Exx_uc.detach(), Eyy_gv.reshape(-1, 1) - Eyy_uc.detach(),
                       Exy_gv.reshape(-1, 1) - Exy_uc.detach()), 0)
rms_uc = torch.sqrt((error_new * error_new).mean())
mae_uc = error_new.abs().mean()



with torch.no_grad():
    csc = 1e6          # converting to mico strain
    cmap = 'RdYlBu'
    f2, ax2 = plt.subplots(3, 3, figsize=(15, 8))
    img1 = ax2[0, 0].pcolor(xv, yv, Exx_gv*csc,cmap=plt.get_cmap(cmap),vmin=-2.35e3, vmax=2.35e3)
    divider = make_axes_locatable(ax2[0,0])
    ct1 = divider.append_axes("right", size="5%", pad=0.05)
    ct1.set_axis_off()
    ax2[0, 0].set_ylabel('$\epsilon_{xx}$',fontsize=30)
    ax2[0, 0].set_title('Saint-Venant', fontsize=30)
    img2 = ax2[1, 0].pcolor(xv, yv, Eyy_gv*csc,cmap=plt.get_cmap(cmap),vmin=-6.5e2, vmax=6.5e2)
    divider = make_axes_locatable(ax2[1,0])
    ct2 = divider.append_axes("right", size="5%", pad=0.05)
    ct2.set_axis_off()
    ax2[1, 0].set_ylabel('$\epsilon_{yy}$',fontsize=30)
    img3 = ax2[2, 0].pcolor(xv, yv, Exy_gv*csc,cmap=plt.get_cmap(cmap),vmin=-5e2, vmax=0)
    divider = make_axes_locatable(ax2[2,0])
    ct3 = divider.append_axes("right", size="5%", pad=0.05)
    ct3.set_axis_off()
    ax2[2, 0].set_ylabel('$\epsilon_{xy}$',fontsize=30)
    ax2[0, 0].set_aspect('equal', 'box')
    ax2[1, 0].set_aspect('equal', 'box')
    ax2[2, 0].set_aspect('equal', 'box')
    ax2[0, 0].set_xticklabels([])
    ax2[0, 0].set_yticklabels([])
    ax2[1, 0].set_xticklabels([])
    ax2[1, 0].set_yticklabels([])
    ax2[2, 0].set_xticklabels([])
    ax2[2, 0].set_yticklabels([])

    ax2[0, 1].pcolor(xv, yv, Exx_p.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-2.35e-3, vmax=2.35e-3)
    divider = make_axes_locatable(ax2[0,1])
    ct4 = divider.append_axes("right", size="5%", pad=0.05)
    ct4.set_axis_off()
    ax2[0, 1].set_title('Our Approach', fontsize=30)
    ax2[1, 1].pcolor(xv, yv, Eyy_p.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-6.5e-4, vmax=6.5e-4)
    divider = make_axes_locatable(ax2[1,1])
    ct5 = divider.append_axes("right", size="5%", pad=0.05)
    ct5.set_axis_off()
    ax2[2, 1].pcolor(xv, yv, Exy_p.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-5e-4, vmax=0)
    divider = make_axes_locatable(ax2[2,1])
    ct6 = divider.append_axes("right", size="5%", pad=0.05)
    ct6.set_axis_off()
    ax2[0, 1].set_aspect('equal', 'box')
    ax2[1, 1].set_aspect('equal', 'box')
    ax2[2, 1].set_aspect('equal', 'box')
    ax2[0, 1].set_xticklabels([])
    ax2[0, 1].set_yticklabels([])
    ax2[1, 1].set_xticklabels([])
    ax2[1, 1].set_yticklabels([])
    ax2[2, 1].set_xticklabels([])
    ax2[2, 1].set_yticklabels([])

    ax2[0, 2].pcolor(xv, yv, Exx_uc.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-2.35e-3, vmax=2.35e-3)
    divider = make_axes_locatable(ax2[0,2])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    f2.colorbar(img1, cax=cax1)
    ax2[0, 2].set_title('Standard NN',fontsize=30)
    ax2[1, 2].pcolor(xv, yv, Eyy_uc.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-6.5e-4, vmax=6.5e-4)
    divider = make_axes_locatable(ax2[1,2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    f2.colorbar(img2, cax=cax2)
    ax2[2, 2].pcolor(xv, yv, Exy_uc.detach().reshape(Exx_gv.size()),cmap=plt.get_cmap(cmap),vmin=-5e-4, vmax=0)
    divider = make_axes_locatable(ax2[2,2])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    f2.colorbar(img3, cax=cax3)
    ax2[0, 2].set_aspect('equal', 'box')
    ax2[1, 2].set_aspect('equal', 'box')
    ax2[2, 2].set_aspect('equal', 'box')
    ax2[0, 2].set_xticklabels([])
    ax2[0, 2].set_yticklabels([])
    ax2[1, 2].set_xticklabels([])
    ax2[1, 2].set_yticklabels([])
    ax2[2, 2].set_xticklabels([])
    ax2[2, 2].set_yticklabels([])
    plt.tight_layout(h_pad=0)
    plt.show()
