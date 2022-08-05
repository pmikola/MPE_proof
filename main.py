from __future__ import print_function
from C import C
from DataGen import Generate
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# ------------------------------- INIT --------------------------
cc = C()
path = 'DATASET/training'
names_field = []
names_struct = []
names_meta = []
plot_flag = 1
show_structure = 1
plot_period = 5000  # ms
grid_size = 250  # 300
nsteps = 500  # 750
save_flag = 0
check_data = 1
data_num = 0
DataNum = 5
frame_interval = 8

Generate(DataNum, frame_interval, plot_period, grid_size, plot_flag, show_structure,save_flag,check_data, nsteps,names_struct,names_field,names_meta,path,data_num)

# np.random.seed(2022)
# np.random.seed(828)

# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# ------------------------------- INIT --------------------------

# Create the dataset
#tensor =
# Create the dataloader
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                         shuffle=True, num_workers=workers)