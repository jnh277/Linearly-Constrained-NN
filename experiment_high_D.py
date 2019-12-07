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
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[150,75],
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
parser.add_argument('--dims', type=int, default=6,
                    help='number of dimensions to run experiment for (default: 3)')

args = parser.parse_args()
args.n_data = 5000
args.display = True
args.show_plot = True
args.epochs = 600
args.batch_size = 250
# args.scheduler = 0

if args.seed >= 0:
    torch.manual_seed(args.seed)

n_data = args.n_data
pin_memory = args.pin_memory
dims = args.dims


def vector_field(xt, a=4.0):
    x = xt.clone()
    d = x.size(1)
    x.requires_grad = True
    # q = torch.exp(-sum(x,1))
    # q.size
    F = a*torch.exp(-x.pow(2).sum(1).sqrt()) * torch.sin(4.0*x.prod(1))
    dF = ag.grad(outputs=F, inputs=x, create_graph=False, grad_outputs=torch.ones(F.size()),
           retain_graph=False, only_inputs=True)[0]
    v = torch.zeros(x.size())
    for i in range(d):
        if i < d-1:
            v[:,i] += dF[:,i+1:].sum(1)
        if i > 0:
            v[:,i] += -dF[:,:i].sum(1)
    return v

def vector_field_7x(xt, a=4.0):
    x = xt.clone()
    d = x.size(1)
    x.requires_grad = True
    # q = torch.exp(-sum(x,1))
    # q.size
    F = a*torch.exp(-x.pow(2).sum(1).sqrt()) * torch.sin(4.0*x.prod(1))
    dF = ag.grad(outputs=F, inputs=x, create_graph=False, grad_outputs=torch.ones(F.size()),
           retain_graph=False, only_inputs=True)[0]
    v = torch.zeros(6,1)
    v[0] = dF[2]
    v[1] = dF[5]
    v[2] = -dF[0]
    v[3] = dF[4]
    v[4] = -dF[3]
    v[5] = -dF[2]
    return v

# set network size
n_in = dims
n_h1 = args.net_hidden_size[0]
n_h2 = args.net_hidden_size[1]
n_o = 1

# two outputs for the unconstrained network
n_o_uc = dims

# model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)

# model = derivnets.DivFree2D(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
#                                          nn.Tanh(),nn.Linear(n_h2,n_o)))
model = derivnets.DivFree(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)

# pregenerate validation data
x_val = torch.rand(5000, dims)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

v_true = vector_field(x_val)
y_val = v_true + 0.1*torch.randn(x_val.size())

# generate training data
x_train = torch.rand(n_data, dims)

v_train = vector_field(x_train)
y_train = v_train + 0.1 * torch.randn(x_train.size())

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x[:, 0])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.x[index, :]
        y = self.y[index, :]

        return x, y

training_set = Dataset(x_train, y_train)

# data loader Parameters
DL_params = {'batch_size': args.batch_size,
             'shuffle': True,
             'num_workers': args.num_workers,
             'pin_memory': pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)

# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
if args.scheduler == 1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=50)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)


def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x_train, y_train in training_generator:
        optimizer.zero_grad()
        (yhat, vhat) = model(x_train)
        loss = criterion(y_train,vhat)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    # with torch.no_grad():
    (yhat, vhat) = model(x_val)
    loss = criterion(y_val, vhat)
    return loss

train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])

if args.display:
    print('Training invariant NN')

for epoch in range(args.epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    if args.scheduler == 1:
        scheduler.step(v_loss)
    else:
        scheduler.step(epoch)   # input epoch for scheduled lr, val_loss for plateau
    val_loss[epoch] = v_loss.detach().numpy()
    if args.display:
        print(args.save_file, 'Invariant NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


# work out the rms error for this one
(f_pred, vhat) = model(x_val)
mse = criterion(v_true,vhat)
rms = torch.sqrt(mse.detach())


# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
if args.scheduler == 1:
    scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=50)
else:
    scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 100, gamma=0.5, last_epoch=-1)

def train_uc(epoch):
    model_uc.train()
    total_loss = 0
    n_batches = 0
    for x_train, y_train in training_generator:
        optimizer_uc.zero_grad()
        vhat = model_uc(x_train)
        loss = criterion(y_train, vhat)
        loss.backward()
        optimizer_uc.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval_uc(epoch):
    model_uc.eval()
    with torch.no_grad():
        (vhat) = model_uc(x_val)
        loss = criterion(y_val, vhat)
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



(vhat_uc) = model_uc(x_val)
mse_uc = criterion(v_true,vhat_uc)
rms_uc = torch.sqrt(mse_uc.detach())




if args.display:
    print('Finished')

if args.show_plot:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 2, figsize=(8, 6))
        # ax.pcolor(xv,yv,f_scalar)

        ax[0].plot(np.log(train_loss))
        ax[0].plot(np.log(val_loss))
        # ax[1].plot(loss_save[1:epoch].log().detach().numpy())
        ax[0].set_xlabel('training epoch')
        ax[0].set_ylabel('log mse val loss')
        ax[0].legend(['training loss', 'val loss'])
        ax[0].set_ylim([-5, 0])


        ax[1].plot(np.log(train_loss_uc))
        ax[1].plot(np.log(val_loss_uc))
        ax[1].set_ylabel('log mse val loss')
        ax[1].set_xlabel('training epoch')
        ax[1].legend(['training loss','val loss'])
        ax[1].set_ylim([-5, 0])
        plt.show()


