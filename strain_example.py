import torch
from matplotlib import pyplot as plt
import derivnets
import torch.nn as nn
from torch.utils import data
import numpy as np

epochs = 300
display = True
n_data = 500
batch_size = 100
num_workers = 2
pin_memory = True
scheduler = 1
save_file = ''
l = 20e-3
h = 10e-3

def strain_field(x,y, P=2e3, E=200e9,l=20e-3,h=10e-3,t=5e-3,nu=0.28):
    I = t*h*h*h/12
    Exx = P/E/I*(l-x)*y
    Eyy = -nu*P/E/I*(l-x)*y
    Exy = -(1+nu)*P/2/E/I*((h/2)*(h/2) - y*y)
    return Exx, Eyy, Exy

# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0, 100.0) * l / 100.0, torch.arange(0.0, 50.0) * h / 50.0-h/2])
(Exx_gv, Eyy_gv, Exy_gv) = strain_field(xv, yv, l=l,h=h)


# set network size
n_in = 2
n_h1 = 100
n_h2 = 50
n_o = 1

# two outputs for the unconstrained network
n_o_uc = 3

model = derivnets.Strain2d(nn.Sequential(nn.Linear(n_in,n_h1),nn.Tanh(),nn.Linear(n_h1,n_h2),
                                         nn.Tanh(),nn.Linear(n_h2,n_o)))

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)


# pregenerate validation data
x_val = torch.cat((l*torch.rand(2000, 1),-h/2+h*torch.rand(2000, 1)),1)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(Exx, Eyy, Exy) = strain_field(x1_val, x2_val, l=l, h=h)
Exx_val = Exx + 1e-4 * torch.randn(x1_val.size())
Eyy_val = Eyy + 1e-4 * torch.randn(x1_val.size())
Exy_val = Exy + 1e-4 * torch.randn(x1_val.size())
E_val = torch.cat((Exx_val, Eyy_val, Exy_val), 1)

# generate training data
x_train = torch.cat((l*torch.rand(n_data, 1),-h/2+h*torch.rand(n_data, 1)),1)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(Exx, Eyy, Exy) = strain_field(x1_train, x2_train, l=l, h=h)
Exx_train = Exx + 1e-4 * torch.randn(x1_train.size())
Eyy_train = Eyy + 1e-4 * torch.randn(x1_train.size())
Exy_train = Exy + 1e-4 * torch.randn(x1_train.size())

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x1, x2, Exx, Eyy, Exy):
        'Initialization'
        self.x1 = x1
        self.x2 = x2
        self.Exx = Exx
        self.Eyy = Eyy
        self.Exy = Exy

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x1 = self.x1[index]
        x2 = self.x2[index]
        Exx = self.Exx[index]
        Eyy = self.Eyy[index]
        Exy = self.Exy[index]

        return x1, x2, Exx, Eyy, Exy

training_set = Dataset(x1_train, x2_train, Exx_train, Eyy_train, Exy_train)

# data loader Parameters
DL_params = {'batch_size': batch_size,
             'shuffle': True,
             'num_workers': num_workers,
             'pin_memory': pin_memory}
training_generator = data.DataLoader(training_set, **DL_params)



# ---------------  Set up and train the constrained model -------------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # these should also be setable parameters
if scheduler == 1:
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
    for x1_train, x2_train, Exx_train, Eyy_train, Exy_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        (Exx, Eyy, Exy) = model(x_train)
        loss = (criterion(Exx_train, Exx) + criterion(Eyy_train, Eyy) + criterion(Exy_train, Exy)) / 3  # divide by 2 as it is a mean
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    return total_loss / n_batches

def eval(epoch):
    model.eval()
    # with torch.no_grad():
    (Exx, Eyy, Exy) = model(x_val)
    loss = (criterion(Exx_val, Exx) + criterion(Eyy_val, Exx) + criterion(Exy_val, Exy)) / 3
    return loss.cpu()


train_loss = np.empty([epochs, 1])
val_loss = np.empty([epochs, 1])

if display:
    print('Training invariant NN')

for epoch in range(epochs):
    train_loss[epoch] = train(epoch).detach().numpy()
    v_loss = eval(epoch)
    if scheduler == 1:
        scheduler.step(v_loss)
    else:
        scheduler.step(epoch)   # input epoch for scheduled lr, val_loss for plateau
    val_loss[epoch] = v_loss.detach().numpy()
    if display:
        print(save_file, 'Invariant NN: epoch: ', epoch, 'training loss ', train_loss[epoch], 'validation loss', val_loss[epoch])


with torch.no_grad():
    # cmap = 'viridis'
    cmap = 'RdYlBu'
    f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
    ax2[0].pcolor(xv, yv, Exx_gv,cmap=plt.get_cmap(cmap))
    ax2[0].set_xlabel('$x_1$')
    ax2[0].set_ylabel('$x_2$')
    ax2[0].set_title('Exx strain')
    ax2[0].plot(x1_train,x2_train,'ok')


    ax2[1].pcolor(xv, yv, Eyy_gv,cmap=plt.get_cmap(cmap))
    ax2[1].set_xlabel('$x_1$')
    ax2[1].set_ylabel('$x_2$')
    ax2[1].set_title('Eyy strain')


    ax2[2].pcolor(xv, yv, Exy_gv,cmap=plt.get_cmap(cmap))
    ax2[2].set_xlabel('$x_1$')
    ax2[2].set_ylabel('$x_2$')
    ax2[2].set_title('Exy Strain')
    plt.show()
