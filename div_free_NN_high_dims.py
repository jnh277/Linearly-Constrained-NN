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
                           help='maximum number of epochs (default: 400)')
parser.add_argument('--seed', type=int, default=-1,
                           help='random seed for number generator (default: -1 means not set)')
parser.add_argument('--batch_size', type=int, default=350,
                           help='batch size (default: 350).')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[1000,500],
                           help='two hidden layer sizes (default: [1000,500]).',)
parser.add_argument('--n_data', type=int, default=10000,
                        help='set number of measurements (default:20000)')
parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loader (default:2)')
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
parser.add_argument('--dims', type=int, default=3,
                    help='number of dimensions (default: 3)')
parser.add_argument('--sigma', type=int, default=1e-2,
                        help='noise standard deviation (default:1e-2)')

args = parser.parse_args()
args.display = True
sigma = args.sigma

if args.seed >= 0:
    torch.manual_seed(1)

n_data = args.n_data
pin_memory = args.pin_memory
dims = args.dims


def vector_field(x, a=25):
    d = x.size(1)
    n = x.size(0)
    w = torch.remainder(torch.linspace(0,d-1,d).unsqueeze(0)*2*3.14/1.61,2*3.14)
    F = a * torch.exp(-3.0 * x.pow(2).sum(1)) * torch.cos(3.0 * x + w).prod(1)
    dF = torch.empty(n, d)

    for i in range(d):
        inds = np.arange(d).tolist()
        inds.pop(i)
        t = a * torch.exp(-3.0 * x.pow(2).sum(1)) * torch.cos(3.0 * x[:, inds] + w[0, inds].unsqueeze(0)).prod(1)
        v = (-3.0 * torch.sin(3.0 * x[:, i] + w[0, i]) - 6.0 * x[:, i] * torch.cos(3.0 * x[:, i] + w[0, i]))
        dF[:, i] = t * v

    v = torch.zeros(n, d)
    for i in range(d):
        if i < d-1:
            v[:,i] += dF[:,i+1:].sum(1)
        if i > 0:
            v[:,i] += -dF[:,:i].sum(1)

    return F, v


# set network size
n_in = dims
n_h1 = args.net_hidden_size[0]
n_h2 = args.net_hidden_size[1]
n_o = 1

# two outputs for the unconstrained network
n_o_uc = dims


# pregenerate validation data
x_val = torch.rand(10000, dims)
(f_true, v_true) = vector_field(x_val)
v_val = v_true + sigma*torch.randn(x_val.size())

# pregenerate training data
x_train = torch.rand(n_data, dims)
(f_t, v_t) = vector_field(x_train)
v_train = v_t + sigma*torch.randn(n_data, dims)

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.x[index, :]
        y = self.y[index, :]

        return x, y

training_set = Dataset(x_train, v_train)

# data loader Parameters
DL_params = {'batch_size': args.batch_size,
             'shuffle': True,
             'num_workers': args.num_workers,
             'pin_memory': pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)

class DivFree(torch.nn.Module):
    def __init__(self, base_net):
        super(DivFree, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        d = x.size(1)
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        v = torch.zeros(x.size())
        for i in range(d):
            if i < d - 1:
                v[:, i] += dydx[:, i + 1:].sum(1)
            if i > 0:
                v[:, i] += -dydx[:, :i].sum(1)
        return y, v


model = DivFree(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))

model_uc = nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),nn.Tanh(),nn.Linear(n_h2,n_o_uc))




# ---------------  Set up and train the uncconstrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
if args.scheduler == 1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-8,
                                                    factor=0.25,
                                                    cooldown=25)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)

def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x_train, y_train in training_generator:
        optimizer.zero_grad()
        (fhat, vhat) = model(x_train)
        loss = criterion(y_train, vhat)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    (fhat, vhat) = model(x_val)
    loss = criterion(v_val, vhat)
    return loss

train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])
learning_rate = np.empty([args.epochs, 1])

if args.display:
    print('Training Constrained NN')

for epoch in range(args.epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    if args.scheduler == 1:
        scheduler.step(v_loss)
    else:
        scheduler.step(epoch)
    val_loss[epoch] = v_loss.detach().numpy()
    for param_group in optimizer.param_groups:
        learning_rate[epoch] = param_group['lr']
    if args.display:
        print(args.save_file, 'Constrained NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])



# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01, weight_decay=1e-3)
if args.scheduler == 1:
    scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-8,
                                                    factor=0.25,
                                                    cooldown=25)
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
        loss = criterion(v_val, vhat)
    return loss.cpu()


train_loss_uc = np.empty([args.epochs, 1])
val_loss_uc = np.empty([args.epochs, 1])
learning_rate_uc = np.empty([args.epochs, 1])

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
    for param_group in optimizer.param_groups:
        learning_rate_uc[epoch] = param_group['lr']
    if args.display:
        print(args.save_file, 'Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])


# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    if args.display:
        print('Saving data')
    data = vars(args)       # puts the config options into a dict
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data['train_loss_uc'] = train_loss_uc
    data['val_loss_uc'] = val_loss_uc
    data['learning_rate'] = learning_rate
    data['learning_rate_uc'] = learning_rate_uc
    sio.savemat('./results/'+ args.save_file+'.mat', data)


if args.display:
    print('Finished')


if args.show_plot:
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
    ax2[0].set_title('output dimension 1')
    ax2[0].legend(['True','learned using cosntrained'])

    # ax2[1].plot(xt[:,0].detach().numpy(),vt[:,dims-1].detach().numpy())
    # ax2[1].plot(xt[:, 0].detach().numpy(), vthat[:, dims-1].detach().numpy())
    ax2[1].plot(xt[:,0].detach().numpy(),vt[:,2].detach().numpy())
    ax2[1].plot(xt[:, 0].detach().numpy(), vthat[:, 2].detach().numpy())
    ax2[0].set_title('output dimension 3')
    ax2[0].legend(['True','learned using cosntrained'])

    ax2[2].plot(xt[:,0].detach().numpy(),ft.detach().numpy())
    ax2[2].plot(xt[:, 0].detach().numpy(), fhat.detach().numpy())
    ax2[2].set_title('Potential field')
    ax2[2].legend(['True','learned using cosntrained'])
    plt.show()

    f, ax = plt.subplots(1,3,figsize=(12,6))
    ax[0].plot(np.log(train_loss))
    ax[0].plot(np.log(val_loss))
    ax[0].set_title('Constrained NN')
    ax[0].legend(['training','validation'])
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('log mean squared error')
    ax[1].plot(np.log(train_loss_uc))
    ax[1].plot(np.log(val_loss_uc))
    ax[1].set_title('Standard NN')
    ax[1].legend(['training','validation'])
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('log mean squared error')
    ax[2].plot(learning_rate)
    ax[2].set_title('Learning rate for constrained NN')
    plt.show()