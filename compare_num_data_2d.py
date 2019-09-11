import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils import data


# compare with same size net as the amount of training data is increased

# training data sizes
# n_data_tests = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
#                 6500, 7000, 7500, 8000]
# n_data_tests = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
n_data_tests = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
# n_data_tests = [1000, 3000]
n_tests = len(n_data_tests)

# net size and layers
n_in = 2
n_h1 = 100
n_h2 = 50
# n_h3 = 50  # not using this layer atm
n_o = 1

n_o_uc = 2  # there are two outputs for the unconstrained case

sc = 1  # scale net size

model_uc = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h1*sc),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h1*sc, n_h2*sc),
    torch.nn.Tanh(),
    torch.nn.Linear(n_h2*sc, n_o_uc),
)

class DerivTanh(torch.nn.Module):
    def __init__(self):
        super(DerivTanh, self).__init__()

    def forward(self, x):
        return 4 / (torch.exp(-x.t()) + torch.exp(x.t())).pow(2)

class DerivRelU(torch.nn.Module):
    def __init__(self):
        super(DerivRelU, self).__init__()

    def forward(self, x):
        tmp = x.t() > 0.0
        return tmp.float()

class QuadReLU(torch.nn.Module):
    def __init__(self):
        super(QuadReLU, self).__init__()

    def forward(self, x):
        tmp = x > 0.0
        return tmp.float() * x**2

class DerivQuadRelU(torch.nn.Module):
    def __init__(self):
        super(DerivQuadRelU, self).__init__()

    def forward(self, x):
        tmp = x.t() > 0.0
        return tmp.float()*2*x.t()


class DerivNet2D(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet2D, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_h1)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_h1, n_h2)
        self.tanh2 = torch.nn.Tanh()
        # self.tanh2 = QuadReLU()
        self.linear3 = torch.nn.Linear(n_h2, n_o)
        self.derivTanh1 = DerivTanh()
        # self.derivTanh2 = DerivQuadRelU()
        self.derivTanh2 = DerivTanh()

    def forward(self, x):
        h1 = self.linear1(x)
        z1 = self.tanh1(h1)
        h2 = self.linear2(z1)
        z2 = self.tanh2(h2)
        y = self.linear3(z2)

        # differential model
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        w3 = self.linear3.weight

        # derivative of h1 with respect to x1 (x-drection)
        dh1dx1 = w1[:,0].unsqueeze(1).repeat(1, nx)

        # derivative of h2 with respect to x2 (y-direction)
        dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)

        dh2dz1 = w2
        dydz2 = w3

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)

        # derivative of output with respect to x1
        dydx1 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()

        # derivative of output with respect to x2
        dydx2 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()

        # print('size: ', dydx.size())
        v1 = dydx2
        v2 = -dydx1
        return (y, v1, v2)

model = DerivNet2D(n_in, n_h1, n_h2, n_o)


def vector_field(x, y, c1x=0.5, c1y=0.8, c2x=0.5, c2y=0.2,l1=0.18,l2=0.18):
    f3_1 = torch.exp(-0.5*(x-c1x).pow(2)/l1**2 - 0.5*(y-c1y).pow(2)/l1**2)
    f3_2 = torch.exp(-0.5 * (c2x-x).pow(2) / l2 ** 2 - 0.5 * (c2y-y).pow(2) / l2 ** 2)
    v1 = -(y-c1y) * f3_1 / l1**2 + (c2y-y)*f3_2/l2**2
    v2 = (x-c1x) * f3_1 / l1**2 - (c2x-x)*f3_2/l2**2
    return (v1, v2)



val_loss_save = torch.empty(n_tests, 1)
val_loss_save_uc = torch.empty(n_tests, 1)


for test in range(n_tests):
    print('test', test)
    del model
    del model_uc
    # reinitialise models
    model = DerivNet2D(n_in, n_h1, n_h2, n_o)

    model_uc = torch.nn.Sequential(
        torch.nn.Linear(n_in, n_h1 * sc),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h1 * sc, n_h2 * sc),
        torch.nn.Tanh(),
        torch.nn.Linear(n_h2 * sc, n_o_uc),
    )

    n_data = n_data_tests[test]
    ## generate data
    x_train = torch.rand(n_data, 2)
    x1_train = x_train[:, 0].unsqueeze(1)
    x2_train = x_train[:, 1].unsqueeze(1)

    (v1, v2) = vector_field(x1_train, x2_train)
    y1_train = v1 + 0.1 * torch.randn(x1_train.size())
    y2_train = v2 + 0.1 * torch.randn(x1_train.size())

    x_val = torch.rand(2000,2)
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

    ## train constrained model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.25, last_epoch=-1)

    train_iters = 125

    for epoch in range(train_iters):
        for x1_train, x2_train, y1_train, y2_train in training_generator:
            optimizer.zero_grad()
            x_train = torch.cat((x1_train, x2_train), 1)
            (yhat, v1hat, v2hat) = model(x_train)
            y_train = torch.cat((y1_train, y2_train), 1)
            v_hat = torch.cat((v1hat, v2hat), 1)
            loss = criterion(y_train, v_hat)
            # print('epoch: ', epoch, ' loss: ', loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step(epoch)

    # compute the final validation cost for this test
    (yhat, v1hat, v2hat) = model(x_val)
    v_hat = torch.cat((v1hat, v2hat), 1)
    val_loss = criterion(y_val, v_hat)
    val_loss_save[test,0] = val_loss
    print('constained NN loss: ', val_loss.item())


    ## train the unconstrained model
    criterion = torch.nn.MSELoss()
    optimizer_uc = torch.optim.Adam(model_uc.parameters(), lr=0.01)
    scheduler_uc = torch.optim.lr_scheduler.StepLR(optimizer_uc, 25, gamma=0.25, last_epoch=-1)


    for epoch in range(train_iters):
        for x1_train, x2_train, y1_train, y2_train in training_generator:
            optimizer_uc.zero_grad()
            x_train = torch.cat((x1_train, x2_train), 1)
            (vhat) = model_uc(x_train)
            y_train = torch.cat((y1_train, y2_train), 1)
            loss = criterion(y_train, vhat)
            # print('epoch: ', epoch, ' loss: ', loss.item())
            loss.backward()
            optimizer_uc.step()
        scheduler_uc.step(epoch)


    # compute teh final validation cost for this test
    (vhat) = model_uc(x_val)
    val_loss = criterion(y_val, vhat)
    val_loss_save_uc[test,0] = val_loss
    print('Unconstrained NN loss: ', val_loss.item())


# plot the true functions
xv, yv = torch.meshgrid([torch.arange(0.0,15.0)/15.0, torch.arange(0.0,15.0)/15.0])
# the scalar potential function

(v1,v2) = vector_field(xv, yv)



# plot the predicted function
x_pred = torch.cat((xv.reshape(15*15,1), yv.reshape(15*15,1)),1)
(f_pred, v1_pred, v2_pred) = model(x_pred)

(v_uc_pred) = model_uc(x_pred)

min_loss = 0.01  # for the value of random noise added

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    # ax.plot(val_loss_save.detach().log().numpy())
    ax.plot(n_data_tests, (val_loss_save.detach()-min_loss).log().numpy())
    ax.plot(n_data_tests, (val_loss_save_uc.detach()-min_loss).log().numpy(), linestyle='--')
    ax.set_ylabel('log(loss-min_loss)')
    ax.set_xlabel('number observations')
    ax.legend(['constrained nn', 'unconstrained nn'])
    plt.show()
