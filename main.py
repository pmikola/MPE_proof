from __future__ import print_function

import os
import time
from GAN import Generator, Discriminator
import matplotlib.cm as cm
import DataGen
from C import C
from DataGen import Generate
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# ------------------------------- INIT --------------------------
cc = C()
path = 'DATASET/training_main'
pathD = 'Models/Dnn/Discriminator.pth'
pathG = 'Models/Gnn/Generator.pth'
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
# number of performed simulations
DataNum = 1
frame_interval = 8
plot_period = 5000  # ms
grid_size = 250  # 300
nsteps = 2

# Datasize of the training dataset
data_size = 500
#number of trained dataset loading laps
laps = 1
# Batch size during training
batch_size = 10

# Number of first layer output channels in Discriminator
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100  # 100
# Size of feature maps in generator
ngf = 250
# Size of feature maps in discriminator
ndf = 250
# Number of training epochs
num_epochs = 25
# Learning rate for optimizers
lr = 0.0001
# Beta1 and beta2 hyperparam for Adam optimizers
beta1 = 0.4
beta2 = 0.4
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Show shapes of Gen and Disc in and out
shape_stat = 0
# Showing samples from training set
show_training_set = False
np.random.seed(2022)
torch.manual_seed(2022)


# ------------------------------- INIT --------------------------
# ------------------------------- FUNC --------------------------
def img_fields(axes, fields_tensor, metas_tensor, grid_size, i):
    ims_fields = axes[i].imshow(np.transpose(
        vutils.make_grid(fields_tensor[i][:, grid_size * n - grid_size: grid_size * n],
                         cmap=cm.hot,
                         extent=[0, grid_size * metas_tensor[i][13], 0, grid_size * metas_tensor[i][13]],
                         padding=2, normalize=True).cpu(), (1, 2, 0)))
    return ims_fields


# ------------------------------- FUNC --------------------------
# ------------------------------- TRAIN -------------------------
for lap_counter in range(0,laps):
    Generate(DataNum, frame_interval, plot_period, grid_size, plot_flag, show_structure, save_flag, check_data, nsteps,
             names_struct, names_field, names_meta, path, check_file_dataset, generate)

    _, names_field, names_struct, names_meta = DataGen.findSubstringInFiles(f, names_struct, names_field, names_meta,
                                                                            save_flag,
                                                                            path)

    structures_tensor = torch.empty((data_size, grid_size, grid_size))
    fields_tensor = torch.empty((data_size, grid_size, grid_size * (nsteps - 1)))
    metas_tensor = torch.empty((len(names_meta), 14))

    # time.sleep(100)
    for i in range(data_size*lap_counter, data_size*(lap_counter+1)):
        Loaded_Structure, Loaded_Field, Loaded_Meta = DataGen.Loader(i, names_struct, names_field, names_meta, path)
        # Loaded_Structure, _, _ = DataGen.Loader(i, names_struct, names_field, names_meta, path)
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
        for i in range(0, 10):
            T = 0
            for n in range(0, nsteps):
                T += 1
                if T % frame_interval == 0:
                    imf0 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 0)
                    imf1 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 1)
                    imf2 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 2)
                    imf3 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 3)
                    imf4 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 4)
                    imf5 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 5)
                    imf6 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 6)
                    imf7 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 7)
                    imf8 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 8)
                    imf9 = img_fields(axes, fields_tensor, metas_tensor, grid_size, 9)
                    ims.append([imf0, imf1, imf2, imf3, imf4, imf5, imf6, imf7, imf8, imf9])
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
            # axes[i].imshow(np.transpose(vutils.make_grid(structures_tensor[i],
            #                                              extent=[0, grid_size * metas_tensor[i][13], 0,
            #                                                      grid_size * metas_tensor[i][13]],
            #                                              padding=2, normalize=True).cpu(), (1, 2, 0)))
            axes[i].imshow(structures_tensor[i])
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

    # if os.path.exists(pathG):
    #     netG.load_state_dict(torch.load(pathG))
    # else:

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

    # if os.path.exists(pathD):
    #     netD.load_state_dict(torch.load(pathD))
    # else:

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize Loss function
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.HingeEmbeddingLoss()
    # Create batch of latent vectors that we will use to visualize
    fixed_noise = torch.randn(batch_size, nz, nz, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    # optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9)
    # optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr, alpha=0.8, eps=1e-09, weight_decay=1e-2, momentum=0.95)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr, alpha=0.8, eps=1e-09, weight_decay=1e-2, momentum=0.95)
    # optimizerD = optim.Adagrad(netD.parameters(), lr=lr, lr_decay=0.1, weight_decay=0.1, initial_accumulator_value=0.1,
    #                            eps=1e-10)
    # optimizerG = optim.Adagrad(netG.parameters(), lr=lr, lr_decay=0.1, weight_decay=0.15, initial_accumulator_value=0.1,
    #                            eps=1e-10)

    ############### Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    img_fakes = []
    azes = []
    iters = 0
    fig_fakes = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(100, 100, wspace=10, hspace=0.6)
    for i in range(0, 10):
        if i < 5:
            az = fig_fakes.add_subplot(grid[0:49, 20 * i:20 * i + 20])
        else:
            az = fig_fakes.add_subplot(grid[51:100, 20 * (i - 5):20 * (i - 5) + 20])
        azes.append(az)
        azes[i].axis("off")

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
            real = data[0].to(device)
            # real = data[0].clone().detach().requires_grad_(True).to(device)
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
                time.sleep(2)
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
                time.sleep(2)
            # Generate fake image batch with G
            fake = netG(noise)
            if shape_stat == 1:
                print(fake.shape)
                print("NetG out - fake gen\n--------------------------\n")
                time.sleep(2)
            # print(fake.shape)
            # time.sleep(4)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            if shape_stat == 1:
                print(output.shape)
                print("NetD out - fake disc\n--------------------------\n")
                time.sleep(2)
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
            if batch_idx % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, batch_idx, len(dataloader_structures),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 5 == 0) or ((epoch == num_epochs - 1) and (batch_idx == len(dataloader_structures) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                fake_learning0 = azes[0].imshow(fake[0])
                fake_learning1 = azes[1].imshow(fake[1])
                fake_learning2 = azes[2].imshow(fake[2])
                fake_learning3 = azes[3].imshow(fake[3])
                fake_learning4 = azes[4].imshow(fake[4])
                fake_learning5 = azes[5].imshow(fake[5])
                fake_learning6 = azes[6].imshow(fake[6])
                fake_learning7 = azes[7].imshow(fake[7])
                fake_learning8 = azes[8].imshow(fake[8])
                fake_learning9 = azes[9].imshow(fake[9])
                # print(fake[0])
                img_fakes.append(
                    [fake_learning0, fake_learning1, fake_learning2, fake_learning3, fake_learning4, fake_learning5,
                     fake_learning6, fake_learning7, fake_learning8, fake_learning9])
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    ani = animation.ArtistAnimation(fig_fakes, img_fakes, interval=30, blit=True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    ############## Save Model After Training ###############
    torch.save(netD.state_dict(), pathD)
    torch.save(netG.state_dict(), pathG)
