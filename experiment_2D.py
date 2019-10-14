import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models
import argparse
import numpy as np
import scipy.io as sio

description = "Train 2D constrained and unconstrained model"

# Arguments that will be saved in config file
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--epochs', type=int, default=300,
                           help='maximum number of epochs (default: 300)')
parser.add_argument('--seed', type=int, default=10,
                           help='random seed for number generator (default: 10)')
parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100).')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[100,50],
                           help='two hidden layer sizes (default: [100,50]).',)
parser.add_argument('--n_data', type=int, default=3000,
                        help='set number of measurements (default:2000)')
parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loader (default:4)')
parser.add_argument('--pin_memory', type=bool, default=False,
                        help='pin memory in data loader (default:False)')
parser.add_argument('--show_plot', type=bool, default=False,
                    help='Enable or disable plotting (default:False)')
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')
# parser.add_argument('--seed', type=int, default=10,
#                            help='random seed for number generator (default: 10)')

args = parser.parse_args()


# torch.manual_seed(args.seed)

n_data = args.n_data


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
             'pin_memory': args.pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)


# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                 min_lr=1e-10,
                                                 factor=0.5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)

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
    with torch.no_grad():
        (yhat, v1hat, v2hat) = model(x_val)
        loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat)) / 2
    return loss


train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])

print('Training invariant NN')
for epoch in range(args.epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    scheduler.step(v_loss)
    val_loss[epoch] = v_loss.detach().numpy()
    print('InvarianT NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])

# work out the rms error for this one
x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
(f_pred, v1_pred, v2_pred) = model(x_pred)
error_new = torch.cat((v1.reshape(400, 1) - v1_pred.detach(), v2.reshape(400, 1) - v2_pred.detach()), 0)
rms_error = torch.sqrt(sum(error_new * error_new) / 800)

# ---------------  Set up and train the uncconstrained model -------------------------------
optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                 min_lr=1e-10,
                                                 factor=0.5)

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

print('Training standard NN')
for epoch in range(args.epochs):
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

error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(), v2.reshape(400) - v2_pred_uc.detach()), 0)
rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    data = vars(args)       # puts the config options into a dict
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data['train_loss_uc'] = train_loss_uc
    data['val_loss_uc'] = val_loss_uc
    data['final_rms_error'] = rms_error
    data['final_rms_error_uc'] = rms_uc
    sio.savemat('./results/' + args.save_file, data)


if args.show_plot:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(np.log(train_loss))
        ax.plot(np.log(val_loss))
        ax.set_xlabel('epoch')
        ax.set_ylabel('log loss')
        ax.legend(['train','val'])
        plt.show()















