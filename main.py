from __future__ import print_function

import time
import matplotlib.cm as cm
import DataGen
from C import C
from DataGen import Generate
# %matplotlib inline
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
from matplotlib import pyplot as plt, animation
import matplotlib.animation as animation
from IPython.display import HTML

# ------------------------------- INIT --------------------------
cc = C()
path = 'DATASET/training'
names_field = []
names_struct = []
names_meta = []
f = []
plot_flag = 0
show_structure = 0
# save the generated data to files
save_flag = 0
# generate dataset
generate = 0
# # check file
check_data = 0
# nuber of the dataset to check
check_file_dataset = 0
# number of sims
DataNum = 1
frame_interval = 8
dataset_size = DataNum
plot_period = 5000  # ms
grid_size = 250  # 300
nsteps = 500  # 750

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
show_training_set = True
# np.random.seed(2022)
# np.random.seed(828)
# torch.manual_seed(2022)
# ------------------------------- INIT --------------------------
# ------------------------------- TRAIN -------------------------


Generate(DataNum, frame_interval, plot_period, grid_size, plot_flag, show_structure, save_flag, check_data, nsteps,
         names_struct, names_field, names_meta, path, check_file_dataset, generate)

_, names_field, names_struct, names_meta = DataGen.findSubstringInFiles(f, names_struct, names_field, names_meta,
                                                                        save_flag,
                                                                        path)

structures_tensor = torch.empty((len(names_struct), grid_size, grid_size))
fields_tensor = torch.empty((len(names_field), grid_size, grid_size * (nsteps - 1)))
metas_tensor = torch.empty((len(names_meta), 14))

# time.sleep(100)
for i in range(0, len(names_struct)):
    Loaded_Structure, Loaded_Field, Loaded_Meta = DataGen.Loader(i, names_struct, names_field, names_meta, path)

    structure_tensor = torch.from_numpy(Loaded_Structure)
    field_tensor = torch.from_numpy(Loaded_Field)
    meta_tensor = torch.from_numpy(Loaded_Meta)
    structures_tensor[i] = structure_tensor
    fields_tensor[i] = field_tensor
    metas_tensor[i] = meta_tensor

# Create the dataloader
dataloader_fields = torch.utils.data.DataLoader(fields_tensor, num_workers=workers, batch_size=batch_size,
                                                shuffle=True, )
dataloader_structures = torch.utils.data.DataLoader(structures_tensor, num_workers=workers, batch_size=batch_size,
                                                    shuffle=True, )
dataloader_metas = torch.utils.data.DataLoader(metas_tensor, num_workers=workers, batch_size=batch_size,
                                               shuffle=True, )
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# Plot some training images

# print(fields_tensor[0][:,0:nsteps].size())
if show_training_set == True:
    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(100, 100, wspace=10, hspace=0.6)
    plt.title("Training Images Discriminator")
    plt.axis("off")
    ims = []
    axes = []
    T = 0
    for i in range(0,len(names_struct)):
        if i < 5:
            ax = fig.add_subplot(grid[0:49, 20*i:20*i+20])
        else:
            ax = fig.add_subplot(grid[51:100, 20 * (i-5):20 * (i-5) + 20])
        axes.append(ax)
        axes[i].axis("off")
        for n in range(0, nsteps):
            T += 1
            if T % frame_interval == 0:
                ims_fields = axes[i].imshow(np.transpose(
                    vutils.make_grid(fields_tensor[i][:, grid_size * n - grid_size: grid_size * n].to(device), cmap=cm.hot,
                                     extent=[0, grid_size * metas_tensor[i][13], 0, grid_size * metas_tensor[i][13]],
                                     padding=2, normalize=True, ).cpu(), (1, 2, 0)))
                ims.append([ims_fields])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
    plt.show()


    fig = plt.figure(figsize=(10, 4))
    plt.title("Training Images Generator")
    del axes
    axes = []
    plt.axis("off")
    for i in range(0,len(names_struct)):
        if i < 5:
            ay = fig.add_subplot(grid[0:49, 20*i:20*i+20])
        else:
            ay = fig.add_subplot(grid[51:100, 20 * (i-5):20 * (i-5) + 20])
        axes.append(ay)
        axes[i].axis("off")
        axes[i].imshow(np.transpose(
                    vutils.make_grid(structures_tensor[i].to(device),
                                     extent=[0, grid_size * metas_tensor[i][13], 0, grid_size * metas_tensor[i][13]],
                                     padding=2, ).cpu(), (1, 2, 0)))
    plt.show()

    print(metas_tensor)
else:
    pass