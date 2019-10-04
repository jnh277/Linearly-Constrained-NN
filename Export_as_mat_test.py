import torch
import numpy as np
import scipy.io as sio


x = torch.linspace(0,6)
y = torch.sin(x)

test = x.numpy()

data = {}       # declare an empty dictionary
data['x'] = x.numpy()
data['y'] = y.numpy()

sio.savemat('./testmat.mat', data)


