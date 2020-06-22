import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models
import argparse
import numpy as np
import derivnets
import torch.nn as nn
import torch.autograd as ag
import scipy.io as sio

description = "Train 2D constrained and unconstrained model"

# Arguments that will be saved in config file
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--epochs', type=int, default=400,
                           help='maximum number of epochs (default: 300)')
parser.add_argument('--seed', type=int, default=-1,
                           help='random seed for number generator (default: -1 means not set)')
parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100).')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[100,50],
                           help='two hidden layer sizes (default: [100,50]).',)
parser.add_argument('--n_data', type=int, default=3000,
                        help='set number of measurements (default:3000)')
parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loader (default:0)')
parser.add_argument('--show_plot', action='store_true',
                    help='Enable plotting (default:False)')
parser.add_argument('--display', action='store_true',
                    help='Enable plotting (default:False)')
parser.add_argument('--save_plot', action='store_true',
                    help='Save plot (requires show_plot) (default:False)')
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')
parser.add_argument('--pin_memory', action='store_true',
                    help='enables pin memory (default:False)')
parser.add_argument('--scheduler', type=int, default=1,
                    help='0 selects interval reduction, 1 selects plateau (default:1)')
parser.add_argument('--constraint_weighting',type=float,default=0.0,
                    help='weighting on constraint cost (default:0.0)')
# for this problem using cuda was slower so have removed teh associated code
# parser.add_argument('--cuda', action='store_true',
#                     help='Enable cuda, will use cuda:0 (default:False)')

# torch.manual_seed(0)



args = parser.parse_args()
args.n_data = 3000
args.display = True
args.show_plot = True
args.constraint_weighting = 0.001

if args.seed >= 0:
    torch.manual_seed(args.seed)

n_data = args.n_data
pin_memory = args.pin_memory
constraint_weighting = args.constraint_weighting

def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


## ------------------ set up models-------------------------- ##
# set network size
n_in = 2
n_h1 = args.net_hidden_size[0]
n_h2 = args.net_hidden_size[1]
n_o = 2         # point observation constraints require 2 output


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

training_set = models.Dataset(x1_train, x2_train, y1_train, y2_train)
#
# data loader Parameters
DL_params = {'batch_size': args.batch_size,
             'shuffle': True,
             'num_workers': args.num_workers,
             'pin_memory': pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)
#
#
# # ---------------  Set up and train the model constrained by 'artificial' observations of constraint -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
if args.scheduler == 1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=15)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)




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
    return (loss, c.pow(2).mean(), loss+constraint_weighting*criterion(torch.zeros(c.size()),c))


train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])
c_train = np.empty([args.epochs, 1])
c_val = np.empty([args.epochs, 1])
#
if args.display:
    print('Training NN with point constraints')

for epoch in range(args.epochs):
    (loss) = train(epoch)
    train_loss[epoch] = loss.detach().numpy()
    # c_train[epoch] = c.detach().numpy()
    (v_loss, c, loss_c) = eval(epoch)
    c_val[epoch] = c.detach().numpy()
    if args.scheduler == 1:
        # scheduler.step(v_loss)
        scheduler.step(loss_c)
    else:
        scheduler.step(epoch)   # input epoch for scheduled lr, val_loss for plateau
    val_loss[epoch] = v_loss.detach().numpy()
    if args.display:
        print(args.save_file, 'point constrained NN: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])




# work out the rms error for this one
x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
(f_pred, c_pred) = model(x_pred)
error_new = v1.reshape(400,1) - f_pred[:,0].detach()
error_new = torch.cat((v1.reshape(400,1) - f_pred[:,0].detach().reshape(400,1), v2.reshape(400,1) - f_pred[:,1].detach().reshape(400,1)), 0)
rms_error = torch.sqrt(sum(error_new * error_new) / 800)
c_mse = c_pred.pow(2).mean()
c_mae = c_pred.abs().mean()
#

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    if args.display:
        print('Saving data')
    data = vars(args)       # puts the config options into a dict
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data['c_val'] = c_val
    data['final_rms_error'] = rms_error.detach().numpy()        # these are tensors so have to convert to numpy
    data['c_mse'] = c_mse
    data['c_mae'] = c_mae
    sio.savemat('./results/'+ args.save_file+'.mat', data)

#
print(rms_error.detach().numpy())
print(c_mae.detach().numpy())
#
if args.display:
    print('Finished')
#
if args.show_plot:
    plt.plot(c_val)
    plt.show()
    plt.plot(val_loss)
    plt.show()
