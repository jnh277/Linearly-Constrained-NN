import torch
from torch.utils import data


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
        self.linear3 = torch.nn.Linear(n_h2, n_o)
        self.derivTanh1 = DerivTanh()
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

        # derivative of h2 with respect to x2 (y-direction) # work out how to use expand instead
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


class DerivNet2D_v2(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet2D_v2, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_h1)
        self.linear1_2 = torch.nn.Linear(n_in, n_h1)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_h1, n_h2)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(n_h2, n_o)
        self.derivTanh1 = DerivTanh()
        self.derivTanh2 = DerivTanh()
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_o = n_o

    def forward(self, x):
        # h1 = self.linear1(x)
        # this is equivlanet to the operation
        # x.mm(w.t()) + b.expand(3, 4)

        # instead we want to introduce a square into this
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        w1 = self.linear1.weight
        w1_2 = self.linear1_2.weight
        w2 = self.linear2.weight
        w3 = self.linear3.weight
        b1 = self.linear1.bias
        b2 = self.linear2.bias
        b3 = self.linear3.bias

        h1 = x.pow(2).mm(w1.t()) + x.mm(w1_2.t()) + b1.expand(nx, self.n_h1)
        z1 = self.tanh1(h1)
        h2 = z1.pow(2).mm(w2.t()) + b2.expand(nx, self.n_h2)
        # h2 = self.linear2(z1)
        z2 = self.tanh2(h2)
        y = z2.pow(2).mm(w3.t()) + b3.expand(nx, self.n_o)
        # y = self.linear3(z2)

        # differential model

        # derivative of h1 with respect to x1 (x-drection)
        # dh1dx1 = w1[:,0].unsqueeze(1).repeat(1, nx)
        dh1dx1 = w1[:,0].unsqueeze(1).expand(self.n_h1, nx) * x[:,0].expand(self.n_h1, nx) + w1_2[:,0].unsqueeze(1).repeat(1, nx)

        # derivative of h2 with respect to x2 (y-direction)
        # dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)
        dh1dx2 = w1[:, 1].unsqueeze(1).expand(self.n_h1, nx) * x[:, 1].expand(self.n_h1, nx) + w1_2[:,1].unsqueeze(1).repeat(1, nx)

        dh2dz1 = 2.0*w2
        dydz2 = 2.0*w3

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)

        # derivative of output with respect to x1
        # test = (dz1dh1 * dh1dx1)
        #
        # print(test.size())
        # print(z1.size())

        # test = z2.t()*(dz2dh2 * dh2dz1.mm(z1*(dz1dh1 * dh1dx1)))
        #
        # print(test.size())
        # print(dydz2.size())

        dydx1 = (dydz2.mm(z2.t()*(dz2dh2 * dh2dz1.mm(z1.t()*(dz1dh1 * dh1dx1))))).t()

        # derivative of output with respect to x2
        dydx2 = (dydz2.mm(z2.t()*(dz2dh2 * dh2dz1.mm(z1.t()*(dz1dh1 * dh1dx2))))).t()

        # print('size: ', dydx.size())
        v1 = dydx2
        v2 = -dydx1
        return (y, v1, v2)

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x1, x2, y1, y2):
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


class DerivNet3D(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2,n_h3, n_o):
        super(DerivNet3D, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_h1)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_h1, n_h2)
        self.tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(n_h2, n_h3)
        self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(n_h3, n_o)
        self.derivTanh1 = DerivTanh()
        self.derivTanh2 = DerivTanh()
        self.derivTanh3 = DerivTanh()

    def forward(self, x):
        h1 = self.linear1(x)
        z1 = self.tanh1(h1)
        h2 = self.linear2(z1)
        z2 = self.tanh2(h2)
        h3 = self.linear3(z2)
        z3 = self.tanh3(h3)
        y = self.linear4(z3)

        # differential model
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        w3 = self.linear3.weight
        w4 = self.linear4.weight

        # derivative of h1 with respect to x1 (x-drection)
        dh1dx1 = w1[:,0].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x2 (y-direction)
        dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x3 (z-direction)
        dh1dx3 = w1[:, 2].unsqueeze(1).repeat(1, nx)

        dh2dz1 = w2
        dh3dz2 = w3
        dydz3 = w4

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)
        dz3dh3 = self.derivTanh2(h3)

        # derivative of output with respect to x1
        # dydx1 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()
        dydx1 = (dydz3.mm(dz3dh3*dh3dz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1)))).t()

        # derivative of output with respect to x2
        # dydx2 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()
        dydx2 = (dydz3.mm(dz3dh3 * dh3dz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2)))).t()

        # derivative of output with respect to x3
        # dydx3 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx3))).t()
        dydx3 = (dydz3.mm(dz3dh3 * dh3dz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx3)))).t()

        # print('size: ', dydx.size())
        return (y, dydx1, dydx2, dydx3)

class DerivNet3D_2layer(torch.nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_o):
        super(DerivNet3D_2layer, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_h1)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(n_h1, n_h2)
        self.tanh2 = torch.nn.Tanh()
        # self.linear3 = torch.nn.Linear(n_h2, n_h3)
        # self.tanh3 = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(n_h2, n_o)
        self.derivTanh1 = DerivTanh()
        self.derivTanh2 = DerivTanh()
        # self.derivTanh3 = DerivTanh()

    def forward(self, x):
        h1 = self.linear1(x)
        z1 = self.tanh1(h1)
        h2 = self.linear2(z1)
        z2 = self.tanh2(h2)
        # h3 = self.linear3(z2)
        # z3 = self.tanh3(h3)
        y = self.linear4(z2)

        # differential model
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        w1 = self.linear1.weight
        w2 = self.linear2.weight
        # w3 = self.linear3.weight
        w4 = self.linear4.weight

        # derivative of h1 with respect to x1 (x-drection)
        dh1dx1 = w1[:,0].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x2 (y-direction)
        dh1dx2 = w1[:,1].unsqueeze(1).repeat(1, nx)

        # derivative of h1 with respect to x3 (z-direction)
        dh1dx3 = w1[:, 2].unsqueeze(1).repeat(1, nx)

        dh2dz1 = w2
        # dh3dz2 = w3
        dydz3 = w4

        # print('size: ', dh2dz1.size())

        dz1dh1 = self.derivTanh1(h1) # this shape means need to do some element wise multiplication
        dz2dh2 = self.derivTanh2(h2)
        # dz3dh3 = self.derivTanh2(h3)

        # derivative of output with respect to x1
        # dydx1 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()
        dydx1 = (dydz3.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx1))).t()

        # derivative of output with respect to x2
        # dydx2 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()
        dydx2 = (dydz3.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx2))).t()

        # derivative of output with respect to x3
        # dydx3 = (dydz2.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx3))).t()
        dydx3 = (dydz3.mm(dz2dh2 * dh2dz1.mm(dz1dh1 * dh1dx3))).t()

        # print('size: ', dydx.size())
        return (y, dydx1, dydx2, dydx3)