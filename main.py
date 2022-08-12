from __future__ import print_function

from matplotlib.artist import Artist
import time
from GAN import Generator, Discriminator
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
from matplotlib import pyplot as plt
import matplotlib.artist as artist
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
DataNum = 10
frame_interval = 8
dataset_size = DataNum
plot_period = 5000  # ms
grid_size = 250  # 300
nsteps = 750

# Discriminator input and Generator output dimensions
x_size, y_size = 250, 250
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 10
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = x_size * y_size
# Number of output channels in Discriminator
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100  # 100
# Size of feature maps in generator
ngf = 250
# Size of feature maps in discriminator
ndf = 250
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Show shapes of Gen and Disc in and out
shape_stat = 0
# Showing samples from training set
show_training_set = False
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

if show_training_set:
    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(100, 100, wspace=10, hspace=0.6)
    plt.title("Training | Fields | Discriminator")
    plt.axis("off")
    ims = []
    axes = []
    # ims_tmp = [fields_tensor[0][:, grid_size - grid_size: grid_size]]*len(names_struct)

    for i in range(0, 10):
        T = 0
        if i < 5:
            ax = fig.add_subplot(grid[0:49, 20 * i:20 * i + 20])
        else:
            ax = fig.add_subplot(grid[51:100, 20 * (i - 5):20 * (i - 5) + 20])
        axes.append(ax)
        axes[i].axis("off")
        for n in range(0, nsteps):
            T += 1
            if T % frame_interval == 0:
                ims_fields = axes[i].imshow(np.transpose(
                    vutils.make_grid(fields_tensor[i][:, grid_size * n - grid_size: grid_size * n],
                                     cmap=cm.hot,
                                     extent=[0, grid_size * metas_tensor[i][13], 0, grid_size * metas_tensor[i][13]],
                                     padding=2, normalize=True).cpu(), (1, 2, 0)))
                ims.append([ims_fields])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    plt.title("Training | Structures | Generator")
    del axes
    axes = []
    plt.axis("off")
    for i in range(0, 10):
        if i < 5:
            ay = fig.add_subplot(grid[0:49, 20 * i:20 * i + 20])
        else:
            ay = fig.add_subplot(grid[51:100, 20 * (i - 5):20 * (i - 5) + 20])
        axes.append(ay)
        axes[i].axis("off")
        axes[i].imshow(np.transpose(vutils.make_grid(structures_tensor[i],
                                                     extent=[0, grid_size * metas_tensor[i][13], 0,
                                                             grid_size * metas_tensor[i][13]],
                                                     padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    print(metas_tensor)
else:
    pass

# Create Train Sets
trainset_fields = torch.utils.data.TensorDataset(torch.FloatTensor(fields_tensor))
trainset_structures = torch.utils.data.TensorDataset(torch.FloatTensor(structures_tensor))
trainset_metas = torch.utils.data.TensorDataset(torch.FloatTensor(metas_tensor))
# print(trainset_fields[0][0].shape)

#
# time.sleep(10)
# Create the dataloader
dataloader_fields = torch.utils.data.DataLoader(trainset_fields, batch_size=batch_size,
                                                shuffle=True)
dataloader_structures = torch.utils.data.DataLoader(trainset_structures, batch_size=batch_size,
                                                    shuffle=True)
dataloader_metas = torch.utils.data.DataLoader(trainset_metas, batch_size=batch_size,
                                               shuffle=True)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()


# custom weights initialization called on netG and netD from random distribution with
# mean = 0 | and stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator
netG = Generator(ngpu, nz, batch_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu, nc, batch_size).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator #64
fixed_noise = torch.randn(batch_size, nz, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

############### Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for batch_idx, data in enumerate(dataloader_structures, 0):
        if shape_stat == 1:
            inputs = data
            inputs = np.array(inputs[0])
            print(inputs.shape)
            print("Input Batch\n--------------------------\n")
        # # Run your training process
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        # real = torch.tensor(data[batch_idx].unsqueeze(dim=2))
        # real = torch.tensor(data[0])
        real = data[0].clone().detach().requires_grad_(True).to(device)
        # real = real.view(-1, x_size * y_size).to(device)

        b_size = real.shape[0]

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        # output = netD(real_cpu).view(-1)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        output = netD(real).view(-1)
        if shape_stat == 1:
            print(output.shape)
            print("Out:NetD - real batch\n--------------------------\n")
            time.sleep(4)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, nz, device=device)
        if shape_stat == 1:
            print(noise.shape)
            print("Random noise\n--------------------------\n")
            time.sleep(4)
        # Generate fake image batch with G
        fake = netG(noise)
        if shape_stat == 1:
            print(fake.shape)
            print("NetG out - fake gen\n--------------------------\n")
            time.sleep(4)
        # print(fake.shape)
        # time.sleep(4)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        if shape_stat == 1:
            print(output.shape)
            print("NetD out - fake disc\n--------------------------\n")
            time.sleep(4)
        # output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader_structures),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 5 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader_structures) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
