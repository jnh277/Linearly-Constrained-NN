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


class DerivNet(torch.nn.Module):
    def __init__(self, *modules):
        super(DerivNet, self).__init__()
        # self.layer = modules[0]
        print(len(modules))
        self.num_layers = len(modules)
        # self.n_layers
       # __class__.__name__ gets names of modules that are input
    def forward(self):
        print(self.layer)
        return 5
