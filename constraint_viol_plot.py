import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models
import argparse
import numpy as np
import scipy.io as sio




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


n_pred = 50
# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, n_pred) * 4.0 / n_pred, torch.arange(0.0, n_pred) * 4.0 / n_pred])
(v1, v2) = vector_field(xv, yv)

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
    f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
    Q = ax2[0].quiver(xv, yv, v1, v2, scale=None, scale_units='inches')
    Q._init()
    assert isinstance(Q.scale, float)
    # ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    ax2[0].set_xlabel('$x_1$')
    ax2[0].set_ylabel('$x_2$')


    ax2[1].pcolor(xv, yv, Cviol, vmin=-6, vmax=6)
    ax2[1].set_xlabel('$x_1$')
    ax2[1].set_ylabel('$x_2$')
    # ax2[1].set_title('Our Approach RMS error ={0:.2f}'.format(rms_new.item()))


    ax2[2].pcolor(xv, yv, Cviol_uc, vmin=-6, vmax=6)
    ax2[2].set_xlabel('$x_1$')
    ax2[2].set_ylabel('$x_2$')
    # ax2[2].set_title('Unconstrained NN RMS error ={0:.2f}'.format(rms_uc.item()))
    plt.show()


















