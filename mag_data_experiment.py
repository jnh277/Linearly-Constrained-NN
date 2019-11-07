import scipy.io as sio
from torch.utils import data
import math
import torch
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import models
import argparse
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--epochs', type=int, default=300,
                           help='maximum number of epochs (default: 300)')
parser.add_argument('--seed', type=int, default=-1,
                           help='random seed for number generator (default: does not set seed)')
parser.add_argument('--batch_size', type=int, default=250,
                           help='batch size (default: 250).')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[150, 75],
                           help='two hidden layer sizes (default: [150, 75]).',)
parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loader (default:4)')
parser.add_argument('--show_plot', action='store_true',
                    help='Enable plotting (default:False)')
# parser.add_argument('--save_plot', action='store_true',
#                     help='Save plot (requires show_plot) (default:False)')
parser.add_argument('--save_file', default='', help='save file name (default: wont save)')
# parser.add_argument('--pin_memory', action='store_true',
#                     help='enables pin memory (default:False)')
parser.add_argument('--scheduler', type=int, default=1,
                    help='0 selects interval reduction, 1 selects plateau (default:1)')
parser.add_argument('--n_train', type=int, default=1000,
                           help='number of data points to use for training (default: 1000).')

args = parser.parse_args()
args.show_plot = 500
args.n_train = 500
args.epochs = 600

if args.seed >= 0:
    torch.manual_seed(args.seed)

# mag_data=sio.loadmat('/Users/johannes/Documents/GitHub/Linearly-Constrained-NN/real_data/magnetic_field_data.mat')
mag_data = sio.loadmat('./real_data/magnetic_field_data.mat')




pos = mag_data['pos']
mag = mag_data['mag']

pos_save = pos.copy()
mag_save = mag.copy()



n = len(pos[:, 0])     # length of data, using all data



# apply a random shuffling to the data
perm = torch.randperm(n)
pos = pos[perm, :]
mag = mag[perm, :]

# Don't normalise inputs as it would change the value of the derivatives and mess up constraint satisfaction
X = torch.from_numpy(pos).float()
y = torch.from_numpy(mag).float()


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, mag):
    'Initialization'
    self.X = X
    self.mag = mag


  def __len__(self):
    'Denotes the total number of samples'
    return len(self.X[:,0])

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    X = self.X[index,:]
    mag = self.mag[index,:]

    return X, mag

# nv = math.floor(n*(1-args.train_split))
# nt = n - nv
nt = min(args.n_train,n)
nv = n - nt


training_set = Dataset(X[0:nt,:], y[0:nt,:])
# data loader Parameters
DL_params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers}
training_generator = data.DataLoader(training_set, **DL_params)

X_val = X[nt:n, :]
mag_val = y[nt:n, :]

## define neural network model
layers = len(args.net_hidden_size)

n_in = 3
n_h1 = args.net_hidden_size[0]
n_h2 = args.net_hidden_size[1]
n_o = 1
n_o_uc = 3
if layers > 2:
    n_h3 = args.net_hidden_size[2]
    model = models.DerivNet3D(n_in, n_h1, n_h2, n_h3, n_o)
else:
    model = models.DerivNet3D_2layer(n_in, n_h1, n_h2, n_o)








## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
if args.scheduler==1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                     min_lr=1e-10,
                                                     factor=0.5,
                                                    cooldown=10)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)


train_loss = np.empty([args.epochs, 1])
val_loss = np.empty([args.epochs, 1])


def train(epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    for x_train, mag_train in training_generator:
        optimizer.zero_grad()
        (yhat, y1hat, y2hat, y3hat) = model(x_train)
        vhat = torch.cat((y1hat, y2hat, y3hat), 1)
        loss = criterion(mag_train, vhat)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss/n_batches

def eval(epoch):
    model.eval()
    with torch.no_grad():
        (yhat, y1hat, y2hat, y3hat) = model(X_val)
        vhat = torch.cat((y1hat, y2hat, y3hat), 1)
        loss = criterion(mag_val, vhat)
    return loss


print('Training constrained NN')
for epoch in range(args.epochs):
    train_loss[epoch, 0] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    if args.scheduler == 1:
        scheduler.step(v_loss)
    else:
        scheduler.step(epoch)
    val_loss[epoch, 0] = v_loss.detach().numpy()
    print('Constrained NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


# Train a standard NN
if layers == 3:
    model_uc = torch.nn.Sequential(
        torch.nn.Linear(n_in, n_h1),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h1, n_h2),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h2, n_h3),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h3, n_o_uc),
    )
elif layers==2:
    model_uc = torch.nn.Sequential(
        torch.nn.Linear(n_in, n_h1),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h1, n_h2),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h2, n_o_uc),
    )

optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
if args.scheduler == 1:
    scheduler_uc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_uc, patience=10,
                                                     min_lr=1e-10,
                                                    factor=0.5,
                                                    cooldown=10)
else:
    scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 50, gamma=0.5, last_epoch=-1)

def train_uc(epoch):
    model_uc.train()
    total_loss = 0
    n_batches = 0
    for x_train, mag_train in training_generator:
        optimizer_uc.zero_grad()
        vhat = model_uc(x_train)
        loss = criterion(mag_train, vhat)
        loss.backward()
        optimizer_uc.step()
        total_loss += loss
        n_batches += 1
    return total_loss/n_batches

def eval_uc(epoch):
    model_uc.eval()
    with torch.no_grad():
        vhat = model_uc(X_val)
        loss = criterion(mag_val, vhat)
    return loss


train_loss_uc = np.empty([args.epochs, 1])
val_loss_uc = np.empty([args.epochs, 1])

print('Training standard NN')
for epoch in range(args.epochs):
    train_loss_uc[epoch, 0] = train_uc(epoch).detach().numpy()
    v_loss = eval_uc(epoch)
    if args.scheduler == 1:
        scheduler_uc.step(v_loss)
    else:
        scheduler_uc.step(epoch)
    val_loss_uc[epoch, 0] = v_loss.detach().numpy()
    print('Standard NN: epoch: ', epoch, 'training loss ', train_loss_uc[epoch], 'validation loss', val_loss_uc[epoch])




#---- see how well it did -------
# generate quiver plot data
grid_x, grid_y= np.meshgrid(np.arange(-1.0, 1.0, 0.2),
                      np.arange(-1.0, 1.0, 0.2))
grid_z = 0.35*np.ones(np.shape(grid_x))

mag_x_interp = griddata(X.numpy(), mag[:,0], (grid_x, grid_y, grid_z), method='linear')
mag_y_interp = griddata(X.numpy(), mag[:,1], (grid_x, grid_y, grid_z), method='linear')

xv = torch.from_numpy(grid_x).float()
yv = torch.from_numpy(grid_y).float()
zv = torch.from_numpy(grid_z).float()

X_pred = torch.cat((xv.reshape(10*10,1), yv.reshape(10*10,1), zv.reshape(10*10,1)),1)
(fpred, f1pred, f2pred, f3pred) = model(X_pred)

fpred_uc = model_uc(X_pred)

with torch.no_grad():
    X_ordered = torch.from_numpy(pos_save).float()
    (p_pred, m1pred, m2pred, m3pred) = model(X)
    fpred_uc = model_uc(X)
    (p_pred_o, m1pred_o, m2pred_o, m3pred_o) = model(X_ordered)
    fpred_uc_o = model_uc(X_ordered)

# ----------------- save configuration options and results -------------------------------
if args.save_file is not '':
    data = vars(args)       # puts the config options into a dict
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    data['train_loss_uc'] = train_loss_uc
    data['val_loss_uc'] = val_loss_uc
    data['p_pred'] = p_pred.numpy()
    data['m1pred'] = m1pred.numpy()
    data['m2pred'] = m2pred.numpy()
    data['m3pred'] = m3pred.numpy()
    data['mpreduc'] = fpred_uc.numpy()
    data['m1pred_o'] = m1pred_o.numpy()
    data['m2pred_o'] = m2pred_o.numpy()
    data['m3pred_o'] = m3pred_o.numpy()
    data['mpreduc_o'] = fpred_uc_o.numpy()
    data['mag_true'] = mag
    data['pos'] = pos
    sio.savemat('./results/'+ args.save_file+'.mat', data)


if args.show_plot:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(2, 2, figsize=(8, 6))
        # ax.pcolor(xv,yv,f_scalar)
        ax[0, 0].quiver(grid_x, grid_y, mag_x_interp, mag_y_interp)
        ax[0, 0].quiver(grid_x, grid_y, f1pred.reshape(10,10).detach(), f2pred.reshape(10,10).detach(),color='r')
        ax[0, 0].set_title('Constrained NN')
        # ax[0].legend(['true','predicted'])

        ax[1, 0].plot(np.log(train_loss))
        ax[1, 0].plot(np.log(val_loss))
        ax[1, 0].set_ylabel('log loss')
        ax[1, 0].set_xlabel('epochs')
        ax[1, 0].legend(['training','validation'])

        ax[0, 1].quiver(grid_x, grid_y, mag_x_interp, mag_y_interp)
        ax[0, 1].quiver(grid_x, grid_y, fpred_uc[:,0].reshape(10,10).detach(), fpred_uc[:,1].reshape(10,10).detach(),color='r')
        ax[0, 1].set_title('Standard NN')
        # ax[0].legend(['true','predicted'])

        ax[1, 1].plot(np.log(train_loss_uc))
        ax[1, 1].plot(np.log(val_loss_uc))
        ax[1, 1].set_ylabel('log loss')
        ax[1, 1].set_xlabel('epochs')
        ax[1, 1].legend(['training','validation'])


        # fig = plt.figure()
        # ax2 = fig.gca(projection='3d')
        # ax2.quiver(X_pred[1:-1:100,0].numpy(),X_pred[1:-1:100,1].numpy(),X[1:-1:100,2].numpy(),m1pred.detach().numpy(),m2pred.detach().numpy(),
        #            m3pred.detach().numpy(), normalize=True)
        # f2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
        # ax2.quiver(X_val[])
        # fig2 = plt.figure()
        # ax2 = plt.axes(projection='3d')
        # ax2.plot3D(pos_save[:,0], pos_save[:,1], pos[:,2], color='blue')
        plt.show()
