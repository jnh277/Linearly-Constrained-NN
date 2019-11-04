import torch
from matplotlib import pyplot as plt
from torch.utils import data
import derivnets
import models
import argparse
import numpy as np
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
parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loader (default:4)')
parser.add_argument('--show_plot', action='store_true',
                    help='Enable plotting (default:False)')
parser.add_argument('--save_plot', action='store_true',
                    help='Save plot (requires show_plot) (default:False)')
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')
parser.add_argument('--pin_memory', action='store_true',
                    help='enables pin memory (default:False)')
parser.add_argument('--scheduler', type=int, default=0,
                    help='0 selects interval reduction, 1 selects plateau (default:0)')
# for this problem using cuda was slower so have removed teh associated code
# parser.add_argument('--cuda', action='store_true',
#                     help='Enable cuda, will use cuda:0 (default:False)')
# parser.add_argument('--seed', type=int, default=10,
#                            help='random seed for number generator (default: 10)')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

n_data = args.n_data
pin_memory = args.pin_memory

def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


## ------------------ set up models-------------------------- ##
# set network size
n_in = 2
n_h1 = args.net_hidden_size[0]
n_h2 = args.net_hidden_size[1]
n_o = 1

# two outputs for the unconstrained network
n_o_uc = 2

# model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)
# model = derivnets.DerivNet(torch.nn.Linear(n_in, n_h1),
#                            torch.nn.Tanh(),
#                            torch.nn.Linear(n_h1, n_h2),
#                            torch.nn.Tanh(),
#                            torch.nn.Linear(n_h2, n_o))

model = derivnets.DerivNet(torch.nn.Linear(n_in, 10),
                           torch.nn.Tanh(),
                           torch.nn.Linear(10, 15),
                           torch.nn.Tanh(),
                           torch.nn.Linear(15, 10),
                           torch.nn.Tanh(),
                           torch.nn.Linear(10,1))

# model_uc = torch.nn.Sequential(
#     torch.nn.Linear(n_in, n_h1),
#     torch.nn.Tanh(),
#     torch.nn.Linear(n_h1, n_h2),
#     torch.nn.Tanh(),
#     torch.nn.Linear(n_h2, n_o_uc),
# )

model_uc = torch.nn.Sequential(torch.nn.Linear(n_in, 10),
                           torch.nn.Tanh(),
                           torch.nn.Linear(10, 15),
                           torch.nn.Tanh(),
                           torch.nn.Linear(15, 10),
                           torch.nn.Tanh(),
                           torch.nn.Linear(10,2))


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
                                                    cooldown=15)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)

def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        (yhat, dydx) = model(x_train)
        v1hat = dydx[1]
        v2hat = -dydx[0]
        loss = (criterion(y1_train, v1hat) + criterion(y2_train, v2hat)) / 2  # divide by 2 as it is a mean
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    with torch.no_grad():
        (yhat, dydx) = model(x_val)
        v1hat = dydx[1]
        v2hat = -dydx[0]
        loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat)) / 2
    return loss.cpu()


train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])

print('Training invariant NN')
for epoch in range(args.epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    if args.scheduler == 1:
        scheduler.step(v_loss)
    else:
        scheduler.step(epoch)   # input epoch for scheduled lr, val_loss for plateau
    val_loss[epoch] = v_loss.detach().numpy()
    print(args.save_file, 'Invariant NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


# work out the rms error for this one
x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
(f_pred, dydx) = model(x_pred)
v1_pred = dydx[1]
v2_pred = -dydx[0]
error_new = torch.cat((v1.reshape(400, 1) - v1_pred.detach(), v2.reshape(400, 1) - v2_pred.detach()), 0)
rms_error = torch.sqrt(sum(error_new * error_new) / 800)

# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
if args.scheduler == 1:
    scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=15)
else:
    scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 100, gamma=0.5, last_epoch=-1)

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


train_loss_uc = np.empty([args.epochs, 1])
val_loss_uc = np.empty([args.epochs, 1])

print('Training standard NN')
for epoch in range(args.epochs):
    train_loss_uc[epoch] = train_uc(epoch).detach().numpy()
    v_loss = eval_uc(epoch)
    if args.scheduler == 1:
        scheduler_uc.step(v_loss)
    else:
        scheduler_uc.step(epoch)
    val_loss_uc[epoch] = v_loss.detach().numpy()
    print(args.save_file, 'Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])

# move model to cpu

# work out final rms error for unconstrainted net
# work out the rms error for this trial
(v_pred_uc) = model_uc(x_pred)
v1_pred_uc = v_pred_uc[:, 0]
v2_pred_uc = v_pred_uc[:, 1]

error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(), v2.reshape(400) - v2_pred_uc.detach()), 0)
rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    data = vars(args)       # puts the config options into a dict
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data['train_loss_uc'] = train_loss_uc
    data['val_loss_uc'] = val_loss_uc
    data['final_rms_error'] = rms_error.detach().numpy()        # these are tensors so have to convert to numpy
    data['final_rms_error_uc'] = rms_uc.detach().numpy()
    sio.savemat('./results/'+ args.save_file+'.mat', data)


if args.show_plot:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(2, 2, figsize=(8, 6))
        # ax.pcolor(xv,yv,f_scalar)
        ax[0, 0].quiver(xv, yv, v1, v2)
        ax[0, 0].quiver(xv, yv, v1_pred.reshape(20, 20).detach(), v2_pred.reshape(20, 20).detach(), color='r')
        ax[0, 0].legend(['true', 'predicted'])
        ax[0, 0].set_title('constrained NN ')

        ax[1, 0].plot(np.log(train_loss))
        ax[1, 0].plot(np.log(val_loss))
        # ax[1].plot(loss_save[1:epoch].log().detach().numpy())
        ax[1, 0].set_xlabel('training epoch')
        ax[1, 0].set_ylabel('log mse val loss')
        ax[1, 0].legend(['training loss', 'val loss'])

        ax[0, 1].quiver(xv, yv, v1, v2)
        ax[0, 1].quiver(xv, yv, v1_pred_uc.reshape(20, 20).detach(), v2_pred_uc.reshape(20, 20).detach(), color='r')
        ax[0, 1].legend(['true', 'predicted'])
        ax[0, 1].set_title('unconstrained NN ')

        ax[1, 1].plot(np.log(train_loss_uc))
        ax[1, 1].plot(np.log(val_loss_uc))
        ax[1, 1].set_ylabel('log mse val loss')
        ax[1, 1].set_xlabel('training epoch')
        ax[1, 1].legend(['training loss','val loss'])

        # Initialize second plot
        f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
        Q = ax2[0].quiver(xv, yv, v1, v2, scale=None, scale_units='inches')
        Q._init()
        assert isinstance(Q.scale, float)
        ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
        ax2[0].set_xlabel('$x_1$')
        ax2[0].set_ylabel('$x_2$')

        error_new = torch.cat((v1.reshape(400, 1) - v1_pred.detach(), v2.reshape(400, 1) - v2_pred.detach()), 0)
        rms_new = torch.sqrt(sum(error_new * error_new) / 800)

        ax2[1].quiver(xv, yv, v1 - v1_pred.reshape(20, 20).detach(), v2 - v2_pred.reshape(20, 20).detach(),
                      scale=Q.scale, scale_units='inches')
        ax2[1].set_xlabel('$x_1$')
        ax2[1].set_ylabel('$x_2$')
        ax2[1].set_title('Our Approach RMS error ={0:.2f}'.format(rms_new.item()))

        error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(), v2.reshape(400) - v2_pred_uc.detach()), 0)
        rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)

        ax2[2].quiver(xv, yv, v1 - v1_pred_uc.reshape(20, 20).detach(), v2 - v2_pred_uc.reshape(20, 20).detach(),
                      scale=Q.scale, scale_units='inches')
        ax2[2].set_xlabel('$x_1$')
        ax2[2].set_ylabel('$x_2$')
        ax2[2].set_title('Unconstrained NN RMS error ={0:.2f}'.format(rms_uc.item()))
        plt.show()
        if args.save_plot:
            f2.savefig('div_free_fields.eps', format='eps')
















