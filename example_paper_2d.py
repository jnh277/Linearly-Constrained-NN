import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models


torch.manual_seed(2)

n_in = 2
n_h1 = 100
n_h2 = 50
n_o = 1

n_o_uc = 2


model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1, n_h2),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2, n_o_uc),
)

# model paper
# f_1(x_1, x_2) = \exp(-ax_1x_2)(ax_1\sin(x_1x_2) - x_1\cos(x_1x_2)),  \ \
#     f_2(x_1, x_2) = \exp(-ax_1x_2)(x_2\cos(x_1x_2) - ax_2\sin(x_1x_2))

def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


## generate data
x_train = 4.0*torch.rand(200,2)
x1_train = x_train[:, 0].unsqueeze(1)
x2_train = x_train[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_train, x2_train)
y1_train = v1 + 0.1 * torch.randn(x1_train.size())
y2_train = v2 + 0.1 * torch.randn(x1_train.size())


x_val = 4.0*torch.rand(2000,2)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_val, x2_val)
y1_val = v1 + 0.1 * torch.randn(x1_val.size())
y2_val = v2 + 0.1 * torch.randn(x1_val.size())
y_val = torch.cat((y1_val, y2_val), 1)

# now put data in a convenient dataset and data loader

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x1,x2,y1,y2):
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

training_set = Dataset(x1_train,x2_train,y1_train,y2_train)

# data loader Parameters
DL_params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 4}
training_generator = data.DataLoader(training_set, **DL_params)

train_iters = 300
## train constrained
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25, last_epoch=-1)


loss_save = torch.empty(train_iters, 1)
val_loss_save = torch.empty(train_iters, 1)
min_val_loss = 1e10

for epoch in range(train_iters):
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)

        (yhat, v1hat, v2hat) = model(x_train)
        loss = (criterion(y1_train, v1hat) + criterion(y2_train, v2hat))/2 # divide by 2 as it is a mean
        loss.backward()
        optimizer.step()
    loss_save[epoch, 0] = loss



    (yhat, v1hat, v2hat) = model(x_val)
    val_loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat))/2 # divide by 2 as it is a mean
    val_loss_save[epoch,0] = val_loss
    scheduler.step(epoch)
    print('epoch: ', epoch, ' val loss: ', val_loss.item())


    if val_loss*1.01 < min_val_loss:
        min_val_loss = val_loss
        last_decrease = epoch
    else:
        if (epoch > 41 + last_decrease) and (epoch > 60):
            break

## train unconstrained
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_uc.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25, last_epoch=-1)


loss_save_uc = torch.empty(train_iters, 1)
val_loss_save_uc = torch.empty(train_iters, 1)
min_val_loss = 1e10


for epoch_uc in range(train_iters):
    for x1_train, x2_train, y1_train, y2_train in training_generator:
        optimizer.zero_grad()
        x_train = torch.cat((x1_train, x2_train), 1)
        vhat = model_uc(x_train)
        y_train = torch.cat((y1_train, y2_train), 1)
        loss = criterion(y_train, vhat)
        loss.backward()
        optimizer.step()
    loss_save_uc[epoch_uc, 0] = loss


    (vhat) = model_uc(x_val)
    val_loss = criterion(y_val, vhat)
    val_loss_save_uc[epoch_uc,0] = val_loss
    scheduler.step(epoch_uc)
    print('epoch: ', epoch_uc, ' val loss: ', val_loss.item())


    if val_loss*1.01 < min_val_loss:
        min_val_loss = val_loss
        last_decrease = epoch_uc
    else:
        if (epoch_uc > 41 + last_decrease) and (epoch_uc > 60):
            break

# plot the true functions
xv, yv = torch.meshgrid([torch.arange(0.0,20.0)*4.0/20.0, torch.arange(0.0,20.0)*4.0/20.0])
# the scalar potential function

(v1,v2) = vector_field(xv, yv)



# plot the predicted function
x_pred = torch.cat((xv.reshape(20*20,1), yv.reshape(20*20,1)),1)
(f_pred, v1_pred, v2_pred) = model(x_pred)
(v_pred_uc) = model_uc(x_pred)
v1_pred_uc = v_pred_uc[:,0]
v2_pred_uc = v_pred_uc[:,1]

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 2, figsize=(8, 6))
    # ax.pcolor(xv,yv,f_scalar)
    ax[0,0].quiver(xv, yv, v1, v2)
    ax[0,0].quiver(xv, yv, v1_pred.reshape(20,20).detach(), v2_pred.reshape(20,20).detach(),color='r')
    ax[0,0].legend(['true','predicted'])
    ax[0,0].set_title('constrained NN ')


    # ax[1].plot(loss_save[1:epoch].log().detach().numpy())
    ax[1,0].plot(val_loss_save[1:epoch].log().detach().numpy(),color='b')
    # ax[1].plot(loss_save[1:epoch].log().detach().numpy())
    ax[1, 0].set_xlabel('training epoch')
    ax[1,0].set_ylabel('log mse val loss')


    ax[0,1].quiver(xv, yv, v1, v2)
    ax[0,1].quiver(xv, yv, v1_pred_uc.reshape(20,20).detach(), v2_pred_uc.reshape(20,20).detach(),color='r')
    ax[0,1].legend(['true','predicted'])
    ax[0,1].set_title('unconstrained NN ')

    ax[1,1].plot(val_loss_save_uc[1:epoch_uc].log().detach().numpy(), color='b')
    ax[1,1].set_ylabel('log mse val loss')
    ax[1, 1].set_xlabel('training epoch')

    # Initialize second plot
    f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
    Q = ax2[0].quiver(xv, yv, v1, v2, scale=None, scale_units='inches')
    Q._init()
    assert isinstance(Q.scale, float)
    ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    ax2[0].set_xlabel('$x_1$')
    ax2[0].set_ylabel('$x_2$')


    error_new = torch.cat((v1.reshape(400,1) - v1_pred.detach(),v2.reshape(400,1) - v2_pred.detach()),0)
    rms_new = torch.sqrt(sum(error_new * error_new) / 800)

    ax2[1].quiver(xv, yv, v1-v1_pred.reshape(20,20).detach(), v2-v2_pred.reshape(20,20).detach(), scale=Q.scale, scale_units='inches')
    ax2[1].set_xlabel('$x_1$')
    ax2[1].set_ylabel('$x_2$')
    ax2[1].set_title('Our Approach RMS error ={0:.2f}'.format(rms_new.item()))

    error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(),v2.reshape(400) - v2_pred_uc.detach()),0)
    rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)

    ax2[2].quiver(xv, yv, v1-v1_pred_uc.reshape(20,20).detach(), v2-v2_pred_uc.reshape(20,20).detach(), scale=Q.scale, scale_units='inches')
    ax2[2].set_xlabel('$x_1$')
    ax2[2].set_ylabel('$x_2$')
    ax2[2].set_title('Unconstrained NN RMS error ={0:.2f}'.format(rms_uc.item()))
    # f2.savefig('div_free_fields.eps', format='eps')
    plt.show()
