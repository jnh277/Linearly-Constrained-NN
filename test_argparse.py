# %% Import and get config
import os
import sys
import json
import argparse
import torch
import datetime
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from warnings import warn

description = "Train model to automatically classify ECG tracings"

# Arguments that will be saved in config file
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--epochs', type=int, default=70,
                           help='maximum number of epochs (default: 70)')
parser.add_argument('--seed', type=int, default=10,
                           help='random seed for number generator (default: 10)')
parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100).')
# parser.add_argument('--lr', type=float, default=0.001,
#                            help='learning rate (default: 0.001)')
# parser.add_argument("--patience", type=int, default=10,
#                            help='maximum number of epochs without reducing the learning rate (default: 10)')
# add_argument("--min_lr", type=float, default=1e-7,
#                            help='minimum learning rate (default: 1e-7)')
# add_argument("--lr_factor", type=float, default=0.1,
#                            help='reducing factor for the lr in a plateu (default: 0.1)')
parser.add_argument('--net_hidden_size', type=int, nargs='+', default=[100,50],
                           help='two hidden layer sizes (default: [100,50]).')


args = parser.parse_args()

print(args.epochs)



