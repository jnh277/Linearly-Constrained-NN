import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models
import argparse
import numpy as np
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable



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

model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)


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


n_pred = 100
# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, n_pred) * 4.0 / n_pred, torch.arange(0.0, n_pred) * 4.0 / n_pred])
(v1, v2) = vector_field(xv, yv)

xv2, yv2 = torch.meshgrid([torch.arange(0.0, 20) * 4.0 / 20, torch.arange(0.0, 20) * 4.0 / 20])
(v1_2, v2_2) = vector_field(xv2, yv2)

# generate training data
n_data = 200

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

model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)
model.load_state_dict(torch.load('constrained_2d.pt'))

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)
model_uc.load_state_dict(torch.load('standard_2d.pt'))

# work out the rms error for this one
x_pred = torch.cat((xv.reshape(n_pred * n_pred, 1), yv.reshape(n_pred * n_pred, 1)), 1)
(f_pred, v1_pred, v2_pred) = model(x_pred)




(v_pred_uc) = model_uc(x_pred)
v1_pred_uc = v_pred_uc[:, 0]
v2_pred_uc = v_pred_uc[:, 1]





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

    print(Cviol.max())

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

    print(Cviol_uc.max())


with torch.no_grad():
    # Initialize second plot
    # f2, ax2 = plt.subplots(1, 3, figsize=(14, 4),subplot_kw={'aspect': 1})
    # Q = ax2[0].quiver(xv2, yv2, v1_2, v2_2, scale=None, scale_units='inches')
    # ax2[0].plot([1.0, 3.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 3.0, 1.0], '--')
    # Q._init()
    # assert isinstance(Q.scale, float)
    # ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    # ax2[0].set_xlabel('$x_1$')
    # ax2[0].set_ylabel('$x_2$')
    # ax2[0].set_aspect('equal', 'box')


    # c1 = ax2[1].pcolor(xv, yv, Cviol, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    # ax2[1].set_xlabel('$x_1$')
    # ax2[1].set_ylabel('$x_2$')
    # f2.colorbar(c1, ax=ax2[1])
    # ax2[1].set_aspect('equal', 'box')
    # # ax2[1].set_title('Our Approach RMS error ={0:.2f}'.format(rms_new.item()))
    #
    # c2 = ax2[2].pcolor(xv, yv, Cviol_uc, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    # ax2[2].set_xlabel('$x_1$')
    # ax2[2].set_ylabel('$x_2$')
    #
    # f2.colorbar(c2, ax=ax2[2])
    # ax2[2].set_aspect('equal', 'box')
    # # ax2[2].set_title('Unconstrained NN RMS error ={0:.2f}'.format(rms_uc.item()))
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

    Q = ax1.quiver(xv2, yv2, v1_2, v2_2, scale=None, scale_units='inches')
    ax1.plot([1.0, 3.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 3.0, 1.0], '--')
    Q._init()
    assert isinstance(Q.scale, float)
    ax1.quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal', 'box')

    img2 = ax2.pcolor(xv, yv, Cviol, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    ax2.set_aspect('equal', 'box')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax2)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Our Approach')

    img3 = ax3.pcolor(xv, yv, Cviol_uc, vmin=-Cviol_uc.max(), vmax=Cviol_uc.max())
    ax3.set_aspect('equal', 'box')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img3, cax=cax3)
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_title('Unconstrained Neural Network')

    plt.tight_layout(h_pad=1)
    plt.show()

    # fig.savefig('constraint_violations.png', format='png')



















