import torch
from matplotlib import pyplot as plt
from torch.utils import data
import argparse
import numpy as np
import derivnets
import torch.nn as nn
import scipy.io as sio
import torch.autograd as ag

description = "Train 2D constrained and unconstrained model"

# Arguments that will be saved in config file
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--epochs', type=int, default=400,
                           help='maximum number of epochs (default: 300)')
parser.add_argument('--seed', type=int, default=-1,
                           help='random seed for number generator (default: -1 means not set)')
parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100).')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[200,100],
                           help='two hidden layer sizes (default: [100,50]).',)
parser.add_argument('--n_data', type=int, default=3000,
                        help='set number of measurements (default:3000)')
parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loader (default:4)')
parser.add_argument('--show_plot', action='store_true',
                    help='Enable plotting (default:False)')
parser.add_argument('--display', action='store_true',
                    help='Enable plotting (default:False)')
parser.add_argument('--save_plot', action='store_true',
                    help='Save plot (requires show_plot) (default:False)')
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')
parser.add_argument('--pin_memory', action='store_true',
                    help='enables pin memory (default:False)')
parser.add_argument('--scheduler', type=int, default=0,
                    help='0 selects interval reduction, 1 selects plateau (default:0)')
parser.add_argument('--dims', type=int, default=7,
                    help='number of dimensions must be 7 (default: 7)')

args = parser.parse_args()
args.n_data = 5000
args.display = True
args.show_plot = True
args.epochs = 300
args.batch_size = 350
sigma = 1e-3
args.scheduler = 1

# if args.seed >= 0:
torch.manual_seed(1)

n_data = args.n_data
pin_memory = args.pin_memory
dims = 6

def vector_field(x, a=50):
    d = x.size(1)
    n = x.size(0)
    w = torch.linspace(0, 3, d).unsqueeze(0)
    F = a*torch.exp(-3.0 * x.pow(2).sum(1)) * torch.cos(3.0 * x + w).prod(1)
    dF = torch.empty(n, d)
    for i in range(d):
        inds = np.arange(d) != i
        t = a * torch.exp(-3.0 * x.pow(2).sum(1))*torch.cos(3.0 * x[:,inds] + w[0,inds].unsqueeze(0)).prod(1)
        v = (-3.0*torch.sin(3.0*x[:,i]+w[0,i])-6.0*x[:, i]*torch.cos(3.0*x[:, i]+w[0,i]))
        dF[:,i] = t*v
    return F, dF


# set network size
n_in = dims
n_h1 = 1000
n_h2 = 500
# n_h3 = 250
n_o = 1

# two outputs for the unconstrained network
n_o_uc = 1


# pregenerate validation data
x_val = torch.rand(10000, dims)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)
(f_true, v_true) = vector_field(x_val)
v_val = v_true + sigma*torch.randn(x_val.size())


model = derivnets.Conservative_7D(nn.Sequential(nn.Linear(n_in,n_h1),
                                                nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))




# ---------------  Set up and train the uncconstrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer_uc = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
if args.scheduler == 1:
    scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-8,
                                                    factor=0.25,
                                                    cooldown=25)
else:
    scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 100, gamma=0.1, last_epoch=-1)

def train_uc(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for i in range(40):
        x_train = -0.1 + 1.2*torch.rand(args.batch_size, dims)
        (f, df) = vector_field(x_train)
        y_train = f+ sigma*torch.randn(f.size())
        optimizer_uc.zero_grad()
        (fhat, vhat) = model(x_train)
        # loss = criterion(y_train.unsqueeze(1), fhat)+ criterion(df, vhat)
        loss = criterion(df, vhat)
        loss.backward()
        optimizer_uc.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval_uc(epoch):
    model.eval()
    # with torch.no_grad():
    (fhat, vhat) = model(x_val)
    # loss = criterion(f_true.unsqueeze(1), fhat)
    loss = criterion(v_true, vhat)
    return loss

train_loss_uc = np.empty([args.epochs, 1])
val_loss_uc = np.empty([args.epochs, 1])

if args.display:
    print('Training standard NN')

for epoch in range(args.epochs):
    train_loss_uc[epoch] = train_uc(epoch).detach().numpy()
    v_loss = eval_uc(epoch)
    if args.scheduler == 1:
        scheduler_uc.step(v_loss)
    else:
        scheduler_uc.step(epoch)
    val_loss_uc[epoch] = v_loss.detach().numpy()
    if args.display:
        print(args.save_file, 'Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])



# (vhat_uc) = model_uc(x_val)
# mse = criterion(v_true,vhat_uc)
# rms_uc = torch.sqrt(mse.detach())


nt = 100
xt = torch.zeros(nt,dims)
xt[:, 0] = torch.linspace(0,1.0,nt)
xt[:, 2] = torch.linspace(0, 1.0, nt)
(ft, vt) = vector_field(xt)
(fhat, vthat) = model(xt)

f2, ax2 = plt.subplots(1, 3, figsize=(12, 6))
# ax.pcolor(xv,yv,f_scalar)

ax2[0].plot(xt[:,0].detach().numpy(),vt[:,0].detach().numpy())
ax2[0].plot(xt[:, 0].detach().numpy(), vthat[:, 0].detach().numpy())

ax2[1].plot(xt[:,0].detach().numpy(),vt[:,dims-1].detach().numpy())
ax2[1].plot(xt[:, 0].detach().numpy(), vthat[:, dims-1].detach().numpy())

ax2[2].plot(xt[:,0].detach().numpy(),ft.detach().numpy())
ax2[2].plot(xt[:, 0].detach().numpy(), fhat.detach().numpy())
plt.show()