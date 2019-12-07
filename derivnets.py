import torch
import math
import torch.autograd as ag
import sys
from torch.utils import data

class DerivNet(torch.nn.Module):
    def __init__(self, base_net):
        super(DerivNet, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        return y, dydx

class DivFree2D(torch.nn.Module):
    def __init__(self, base_net):
        super(DivFree2D, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        return y, dydx[:,1].unsqueeze(1), -dydx[:,0].unsqueeze(1)


class Strain2d(torch.nn.Module):
    def __init__(self, base_net, nu=0.28):
        super(Strain2d, self).__init__()
        self.base_net = base_net
        self.nu = nu

    def forward(self, x):
        x.requires_grad = True
        y = self.base_net(x)
        g = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        hx = ag.grad(outputs=g[:, 0], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:, 0].size()),
                       retain_graph=True, only_inputs=True)[0]
        hy = ag.grad(outputs=g[:, 1], inputs=x, create_graph=True, grad_outputs=torch.ones(g[:, 1].size()),
                     retain_graph=True, only_inputs=True)[0]
        Exx = hy[:, 1].unsqueeze(1) - self.nu * hx[:, 0].unsqueeze(1)
        Eyy = hx[:, 0].unsqueeze(1) - self.nu * hy[:, 1].unsqueeze(1)
        Exy = - (1 + self.nu) * hx[:, 1].unsqueeze(1)
        return Exx, Eyy, Exy

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



class DivFree_7x(torch.nn.Module):
    def __init__(self, base_net):
        super(DivFree_7x, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        d = x.size(1)
        n = x.size(0)
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]
        v = torch.empty(6,1)
        v[:,0] = dydx[:,2]
        v[:, 1] = dydx[:, 5]
        v[:, 2] = -dydx[:, 0]
        v[:, 3] = dydx[:, 4]
        v[:, 4] = -dydx[:, 3]
        v[:, 5] = -dydx[:, 2]
        return y, v

class Conservative_7D(torch.nn.Module):
    def __init__(self, base_net):
        super(Conservative_7D, self).__init__()
        self.base_net = base_net

    def forward(self, x):
        d = x.size(1)
        n = x.size(0)
        x.requires_grad = True
        y = self.base_net(x)
        dydx = ag.grad(outputs=y, inputs=x, create_graph=True, grad_outputs=torch.ones(y.size()),
                       retain_graph=True, only_inputs=True)[0]

        return y, dydx


class DerivTanh(torch.nn.Module):
    def __init__(self):
        super(DerivTanh, self).__init__()

    def forward(self, x):
        return 4 / (torch.exp(-x.t()) + torch.exp(x.t())).pow(2)

class DerivReLU(torch.nn.Module):
    def __init__(self):
        super(DerivReLU, self).__init__()

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


class DerivNet_old(torch.nn.Module):
    def __init__(self, *modules):
        super(DerivNet_old, self).__init__()
        # *modules is a tuple, I would say its fine to leave it like that
        # self.layer = modules[0]
        self.supported = ['Linear', 'Tanh', 'ReLU']
        self.num_layers = len(modules)

        self.layer_names = []
        for i in range(self.num_layers):
            self.layer_names.append(modules[i].__class__.__name__)
            self.add_module(str(i), modules[i])     # can then access inside the self._modules[] dict

        # check if layer types are supported
        assert all(elem in self.supported for elem in self.layer_names) , 'DerivNet currently only supports '+' '.join(self.supported)
        # odd layers should be linear
        assert all(item=='Linear' for item in self.layer_names[::2]), 'All odd layers should be linear'
        # even layers should be activation functions
        assert all(item != 'Linear' for item in self.layer_names[1::2]), 'All even layers should be activation functions'

        # get input dimensions
        self.dim_x = self._modules['0'].in_features

        # initialise the layer derivatives
        self.weights = []                       # store all the weights in an empty list
        self.dfcn = []        # add modules for derivatives of activation layers
        for i in range(self.num_layers):
            if not (i % 2):           # if a linear layer
                self.weights.append(self._modules[str(i)].weight)
            else:   # for activation layers append the derivative function
                if self.layer_names[i] == 'Tanh':
                    self.dfcn.append(DerivTanh())
                elif self.layer_names[i] == 'ReLU':
                    self.dfcn.append(DerivReLU)


    def forward(self, x):
        # print(self.layer)

        dh1dx = []
        (nx, dx) = x.size()  # nx is number of data points, dx is data dimension (must match n_in)
        assert dx == self.dim_x

        # propagate the potential function
        h = []
        z = []
        for i in range(math.ceil(self.num_layers/2)):
            if i == 0:
                h.append(self._modules[str(2*i)](x))
            else:
                h.append(self._modules[str(2*i)](z[i-1]))
            if 2*i+1 < self.num_layers:
                z.append(self._modules[str(2*i+1)](h[i]))

        # for i in range(self.dim_x):
        #     dh1dx.append(self.weights[:, i].unsqueeze(1).repeat(1, nx))

        if self.num_layers % 2:     # if final layer was a linear one
            y = h[-1]
        else:
            y = z[-1]               # if final layer was activation function

        # now do derivatives
        dzdh = []
        for i in range(len(self.dfcn)):
            dzdh.append(self.dfcn[i](h[i]))
        dydx = []
        for k in range(self.dim_x):
            for i in range(math.ceil(self.num_layers/2)):
                if i == 0:
                    tmp = self.weights[0][:,k].unsqueeze(1).repeat(1, nx)
                else:
                    tmp = self.weights[i].mm(tmp)
                if 2*i+1 < self.num_layers:
                    tmp = dzdh[i] * tmp
            dydx.append(tmp.t())

        return y, dydx
