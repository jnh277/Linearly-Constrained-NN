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
# This code runs a comparison of our proposed method and the approach of including
# a finite number of point evaluations of the constraint into the cost function as
# discussed in Section 5.2


import torch
from matplotlib import pyplot as plt
from torch.utils import data
import numpy as np
import torch.nn as nn
import torch.autograd as ag


description = 'Trains a 2D constrained and point constrained models'


n_data = 3000
pin_memory = False
constraint_weighting = 0.32
epochs = 400

def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


## ------------------ set up models-------------------------- ##
# set network size
n_in = 2
n_h1 = 100
n_h2 = 50
n_o = 2         # point observation constraints require 2 output



# define model taht allows for penalising point evaluation of the constraint
class pointObsDivFree2D(torch.nn.Module):
    def __init__(self, base_net):
        super(pointObsDivFree2D, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        y1 = y[:,0]
        y2 = y[:,1]
        dy1dx = ag.grad(outputs=y1, inputs=x, create_graph=True, grad_outputs=torch.ones(y1.size()),
                       retain_graph=True, only_inputs=True)[0]
        dy2dx = ag.grad(outputs=y2, inputs=x, create_graph=True, grad_outputs=torch.ones(y2.size()),
                       retain_graph=True, only_inputs=True)[0]
        # return y, dydx[:,1].unsqueeze(1)+dydx[:,0].unsqueeze(1)
        # the constraint we are trying to satisfy is dy1/dx1 + dy2/dx2 = 0
        c = dy1dx[:, 0] + dy2dx[:, 1]
        return y, c

model = pointObsDivFree2D(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))



###      define model for our proposed appraoch
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


model_constrained = DivFree2D(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,1)))


# pregenerate validation data
x_val = 4.0 * torch.rand(2000, 2)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_val, x2_val)
y1_val = v1 + 0.1 * torch.randn(x1_val.size())
y2_val = v2 + 0.1 * torch.randn(x1_val.size())
y_val = torch.cat((y1_val, y2_val), 1)


# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, 20.0) * 4.0 / 20.0, torch.arange(0.0, 20.0) * 4.0 / 20.0])
(v1, v2) = vector_field(xv, yv)

# generate training data
x_train = 4.0 * torch.rand(n_data, 2)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(v1_t, v2_t) = vector_field(x1_train, x2_train)
y1_train = v1_t + 0.1 * torch.randn(x1_train.size())
y2_train = v2_t + 0.1 * torch.randn(x1_train.size())

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
             'num_workers': 0,
             'pin_memory': pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)
#
#
# # ---------------  Set up and train the model constrained by 'artificial' observations of constraint -------------------------------
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
        (yhat, c) = model(x_train)
        y_train = torch.cat((y1_train, y2_train), 1)
        loss = criterion(y_train, yhat) + constraint_weighting*criterion(torch.zeros(c.size()),c)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return (total_loss / n_batches)

def eval(epoch):
    model.eval()
    (yhat, c) = model(x_val)
    loss = criterion(y_val, yhat)
    return (loss, c.abs().mean(), loss+constraint_weighting*criterion(torch.zeros(c.size()),c))


train_loss = np.empty([epochs, 1])
val_loss = np.empty([epochs, 1])
c_train = np.empty([epochs, 1])
c_val = np.empty([epochs, 1])
#

print('Training NN with point constraints')

for epoch in range(epochs):
    (loss) = train(epoch)
    train_loss[epoch] = loss.detach().numpy()
    # c_train[epoch] = c.detach().numpy()
    (v_loss, c, loss_c) = eval(epoch)
    c_val[epoch] = c.detach().numpy()
    scheduler.step(loss_c)
    val_loss[epoch] = v_loss.detach().numpy()
    print('point constrained NN: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])




# work out the rms error for this model and the constraint violations
x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
(f_pred, c_pred) = model(x_pred)
error_new = v1.reshape(400,1) - f_pred[:,0].detach()
error_new = torch.cat((v1.reshape(400,1) - f_pred[:,0].detach().reshape(400,1), v2.reshape(400,1) - f_pred[:,1].detach().reshape(400,1)), 0)
rms_error = torch.sqrt(sum(error_new * error_new) / 800)
c_mse = c_pred.pow(2).mean()
c_mae = c_pred.abs().mean()


# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer_c = torch.optim.Adam(model_constrained.parameters(), lr=0.01)   # these should also be setable parameters
scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=15)

def train_c(epoch):
    model_constrained.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer_c.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        (yhat, v1hat, v2hat) = model_constrained(x_train)
        loss = (criterion(y1_train, v1hat) + criterion(y2_train, v2hat)) / 2  # divide by 2 as it is a mean
        loss.backward()
        # print(loss.detach().numpy())
        optimizer_c.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval_c(epoch):
    model_constrained.eval()
    # with torch.no_grad():
    (yhat, v1hat, v2hat) = model_constrained(x_val)
    loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat)) / 2
    return loss


train_loss_c = np.empty([epochs, 1])
val_loss_c = np.empty([epochs, 1])

print('Training Constrained NN')

for epoch in range(epochs):
    train_loss_c[epoch] = train_c(epoch).detach().numpy()
    v_loss = eval_c(epoch)
    scheduler_c.step(v_loss)
    val_loss_c[epoch] = v_loss.detach().numpy()
    print('Constrained NN: epoch: ', epoch, 'training loss ', train_loss_c[epoch], 'validation loss', val_loss_c[epoch])

c_loss_constrained = 0*val_loss_c
# work out the rms error and constraint violations
x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
(f_pred, v1_pred, v2_pred) = model_constrained(x_pred)
error_new_c = torch.cat((v1.reshape(400, 1) - v1_pred.detach(), v2.reshape(400, 1) - v2_pred.detach()), 0)
rms_error_c = torch.sqrt(sum(error_new_c * error_new_c) / 800)
c_loss_constrained = 0*val_loss_c
c_mae_constrained = np.mean(0*val_loss_c)    # this model type is guaranteed to satisfy the constraints so no need to calculate

#
print('')
print('')
print('####------------------------ Results ------------------------####')
print('Proposed constrained approach: RMSE = ', rms_error_c.detach().numpy()[0])
print('Proposed constrained approach:  Mean absolute constraint violation = ',c_mae_constrained)

print('Point constraint approach: RMSE = ', rms_error.detach().numpy()[0])
print('Point constraint approach:  Mean absolute constraint violation = ',c_mae.detach().numpy())
#


plt.plot(c_val)
plt.plot(c_loss_constrained,'--')
plt.xlabel('Epoch')
plt.ylabel('Mean absolute constraint violation')
plt.legend(('Point constraint approach','Our proposed approach'))
plt.show()
plt.plot(np.log(val_loss))
plt.plot(np.log(val_loss_c),'--')
plt.xlabel('Epoch')
plt.ylabel('Log validation loss')
plt.show()
