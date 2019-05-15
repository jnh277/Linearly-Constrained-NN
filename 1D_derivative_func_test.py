import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt



# Training data is 1000 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100).unsqueeze(1)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size())*0.1 + 1.0




val_x = torch.linspace(0, 1, 100).unsqueeze(1)
# True function is sin(2*pi*x) with Gaussian noise
val_y = torch.sin(val_x * (2 * math.pi)) + torch.randn(val_x.size())*0.1 + 1.0

n_in = 1
n_h1 = 100
n_h2 = 50
n_h3 = 50
n_o = 1


model = nn.Sequential(nn.Linear(n_in, n_h1),
                      nn.Tanh(),
                      nn.Linear(n_h1, n_h2),
                      nn.Tanh(),
                      nn.Linear(n_h2, n_h3),
                      nn.Tanh(),
                      nn.Linear(n_h3, n_o)
                      )

## train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1)

train_iters = 10000
loss_save = torch.empty(train_iters, 1)
val_loss_save = torch.empty(train_iters, 1)
for epoch in range(train_iters):
    train_x = torch.linspace(0, 1, 100).unsqueeze(1)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.1 + 1.0

    optimizer.zero_grad()
    yhat = model(train_x)
    loss = criterion(train_y, yhat)
    yhat_val = model(val_x)
    val_loss = criterion(val_y, yhat_val)
    print('epoch: ', epoch, ' loss: ', loss.item(), 'val loss: ', val_loss.item())
    loss.backward()
    optimizer.step()
    loss_save[epoch, 0] = loss
    val_loss_save[epoch, 0] = val_loss
    # scheduler.step(epoch)



test_x = torch.linspace(0, 1, 51).unsqueeze(1)
# test_x.grad.data.zero_() # ensure the gradient is zero
with torch.no_grad():
    prediction = model(test_x)

df = torch.empty(test_x.size())
# now work out the gradient
for i in range(len(test_x)):
    x_p = torch.tensor([test_x[i]], requires_grad=True)
    if type(x_p.grad) is not type(None):
        x_p.grad.data.zero_()               # ensure grad is zero
    f_p = model(x_p)
    f_p.backward()
    df[i] = x_p.grad.item()


df_true = (2 * math.pi)*torch.cos(test_x * (2 * math.pi))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(2, 1, figsize=(4, 6))

    # Plot training data as black stars
    ax[0].plot(train_x.numpy(), train_y.numpy(), 'k*')
    # # Plot predictive means as blue line

    ax[0].plot(test_x.numpy(), prediction.detach().numpy(), 'g')

    ax[0].set_ylim([-3, 3])
    ax[0].legend(['Observed Data', 'trained on non-derivative'])

    ax[1].plot(test_x.numpy(), df.numpy())
    ax[1].plot(test_x.numpy(), df_true.numpy())
    ax[1].legend(['True derivative', 'predicted derivative'])
    # ax[1].plot(loss_save.detach().log().numpy())
    # ax[1].plot(val_loss_save.log().detach().numpy(), 'r')
    # ax[1].legend(['training loss', 'val loss'])
    plt.show()