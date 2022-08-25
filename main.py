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
import torch.nn.functional as F

# ------------------------------- INIT --------------------------

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(torch.cuda.get_device_name(0))
torch.cuda.empty_cache()

# TODO : Speeding up calculationss
# torch.backends.cudnn.benchmark = True
# scaler = torch.cuda.amp.GradScaler()
cc = C()
path = 'DATASET/training_main'
pathD = 'Models/Dnn/Discriminator.pth'
pathG = 'Models/Gnn/Generator.pth'
names_field = []
names_struct = []
names_meta = []
f = []
img_list = []
G_losses = []
D_losses = []
y_top = 0.
img = []
azes = []
plot_flag = 0
show_structure = 0
m = 0
# save the generated data to files
save_flag = m
# generate dataset
generate = m
# # check file
check_data = 0
# nuber of the dataset to check
check_file_dataset = 0
# number of performed simulations
DataNum = 10
frame_interval = 8
plot_period = 5000  # ms
# grid_size / dimension
grid_size = 250  # 300
nsteps = 2

# Datanumber of the training dataset
data_size = 250
# number of trained dataset loading laps
laps = 1
# Batch size during training
batch_size =25

# Displaying progress of the traing - modes of fake generator images + loss plots
disp_progrss = 0
# Number of first layer channels in Discriminator (for rgb is 3 but for our 0,1 data is 1)
num_of_chanells = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100  # 100
# Range of the latent vector values
r_max = 1
r_min = -r_max

# Size of feature maps in generator
features_generator = batch_size
# Size of feature maps in discriminator
features_discriminator = batch_size
# Number of training epochs
num_epochs = 50
# Learning rate for optimizers
lr = 0.0002
# Beta1 and beta2 hyperparam for Adam optimizers
beta1 = 0.95
beta2 = 0.999
# momentum for RMSprop optimizers
momentumG = 0.95
momentumD = 0.95
# Weight clipping values (1 means no clipping)
clip = 0.01
# Show shapes of Gen and Disc in and out
shape_stat = 0
# Showing samples from training set
show_training_set = False
# deleting the models
delate_models = True
delate_G = False
delate_D = False
# Seed
np.random.seed(2022)
torch.manual_seed(2022)

test_data = np.zeros((batch_size, batch_size, grid_size, grid_size))
confidences = torch.zeros((batch_size, batch_size))
for i in range(batch_size + 1):
    a, b, c = np.mgrid[-r_max + i * r_max / batch_size:0 + i * r_max / batch_size:(batch_size * 1j),
              -r_max + i * r_max / batch_size:0 + i * r_max / batch_size:(grid_size * 1j),
              -r_max + i * r_max / batch_size:0 + i * r_max / batch_size:(grid_size * 1j)]
    # print(a)
    test_data[i - 1] = a
    # time.sleep(1)
# ------------------------------- INIT --------------------------
# ------------------------------- FUNC --------------------------
def img_fields(axes, fields_tensor, metas_tensor, grid_size, i):
    ims_fields = axes[i].imshow(np.transpose(
        vutils.make_grid(fields_tensor[i][:, grid_size * n - grid_size: grid_size * n],
                         cmap=cm.hot,
                         extent=[0, grid_size * metas_tensor[i][13], 0, grid_size * metas_tensor[i][13]],
                         padding=2, normalize=True).cpu(), (1, 2, 0)))
    return ims_fields


# custom weights initialization called on netG and netD from random distribution with
# mean = 0 | and stdev = 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------- FUNC --------------------------
# ------------------------------- TRAIN -------------------------
for lap_counter in range(0, laps):
    print('LAP number : ', lap_counter + 1)
    Generate(DataNum, frame_interval, plot_period, grid_size, plot_flag, show_structure, save_flag, check_data, nsteps,
             names_struct, names_field, names_meta, path, check_file_dataset, generate)

    _, names_field, names_struct, names_meta = DataGen.findSubstringInFiles(f, names_struct, names_field, names_meta,
                                                                            save_flag,
                                                                            path)

    structures_tensor = torch.empty((data_size, grid_size, grid_size))
    fields_tensor = torch.empty((data_size, grid_size, grid_size * (nsteps - 1)))
    metas_tensor = torch.empty((len(names_meta), 14))

    # time.sleep(100)
    for i in range(0, data_size):
        Loaded_Structure, Loaded_Field, Loaded_Meta = DataGen.Loader((data_size * lap_counter) + i, names_struct,
                                                                     names_field, names_meta, path)
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
                                                    shuffle=True, pin_memory=True)
    dataloader_structures = torch.utils.data.DataLoader(trainset_structures, batch_size=batch_size,
                                                        shuffle=True, pin_memory=True)
    dataloader_metas = torch.utils.data.DataLoader(trainset_metas, batch_size=batch_size,
                                                   shuffle=True, pin_memory=True)

    # Create the generator
    netG = Generator(ngpu, nz, num_of_chanells, features_generator,grid_size).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    if os.path.exists(pathG):
        if delate_models or delate_G:
            os.remove(pathG)
        else:
            pass
        try:
            netG.load_state_dict(torch.load(pathG))
        except:
            # Apply the weights_init function to randomly initialize all weights
            #  to mean=0, stdev=0.02.
            netG.apply(weights_init)
    else:
        if laps == 1:
            # Apply the weights_init function to randomly initialize all weights
            #  to mean=0, stdev=0.02.
            netG.apply(weights_init)
        else:
            pass

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, num_of_chanells, features_discriminator).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    if os.path.exists(pathD):
        if delate_models or delate_D:
            os.remove(pathD)
            delate_models = False
        else:
            pass
        try:
            netD.load_state_dict(torch.load(pathD))
        except:
            if laps == 1:
                # Apply the weights_init function to randomly initialize all weights
                #  to mean=0, stdev=0.2.
                netD.apply(weights_init)
            else:
                pass
    else:
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)

    # Print the model
    print(netD)
    print("Number of parameters | Discriminator | ", count_parameters(netD))
    time.sleep(1)
    print("Number of parameters | Generator | ", count_parameters(netG))
    time.sleep(1)
    # Initialize Loss function
    criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.1)
    # criterion = nn.BCEWithLogitsLoss()
    # Fixed noise visualisation check
    fixed_noise = torch.reshape(torch.squeeze(torch.randn(batch_size, nz, 1)), (batch_size, nz, 1, 1)).to(device)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    #
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-09, weight_decay=0, amsgrad=True)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-09, weight_decay=0,
                            amsgrad=True)
    # optimizerD = optim.RMSprop(netD.parameters(), lr=lr, alpha=0.9, eps=1e-08, weight_decay=0, momentum=momentumD)
    # optimizerG = optim.RMSprop(netG.parameters(), lr=lr, alpha=0.9, eps=1e-08, weight_decay=0, momentum=momentumG)

    iters = 0
    if lap_counter == 0:
        fakes_loss_modes = plt.figure(figsize=(18, 8))
        grid = plt.GridSpec(100, 100, wspace=0.5, hspace=0.5)
        for i in range(0, 10):
            if i < 5:
                az = fakes_loss_modes.add_subplot(grid[0:23, 10 * i:10 * i + 10])
            if 5 <= i < 10:
                az = fakes_loss_modes.add_subplot(grid[25:48, 10 * (i - 5):10 * (i - 5) + 10])
            # if 10 <= i < 15:
            #     az = fakes_loss_modes.add_subplot(grid[50:73, 10 * (i - 10):10 * (i - 10) + 10])
            # if 15 <= i < 20:
            #     az = fakes_loss_modes.add_subplot(grid[75:98, 10 * (i - 15):10 * (i - 15) + 10])
            else:
                pass
            azes.append(az)
            azes[i].axis("off")
        az = fakes_loss_modes.add_subplot(grid[0:45, 55:100])
        azes.append(az)
        az = fakes_loss_modes.add_subplot(grid[55:100, 55:100])
        azes.append(az)
        az = fakes_loss_modes.add_subplot(grid[55:100, 0:50])
        azes.append(az)
        azes[12].axis("off")
        azes[12].set_title("Discriminator Boundary")
    print("Starting Training Loop...")

    # For each epoch
    z = 0
    window = 3

    for epoch in range(num_epochs):
        if z < window:
            netG.train(True)
            netD.train(False)
        else:
            netG.train(False)
            netD.train(True)
        if z > window*2:
            z = 0
        z += 1
        #print(z)
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader_structures, 0):
            if shape_stat == 1:
                inputs = data
                inputs = np.array(inputs[0])
                print(inputs.shape)
                print("Input Batch\n--------------------------\n")
            ############################
            # (1) Maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad(set_to_none=True)
            # netG.zero_grad(set_to_none=True)
            # Format batch
            real = torch.reshape(torch.squeeze(data[0]), (batch_size, 1, grid_size, grid_size)).to(device)
            # real = data[0].clone().detach().requires_grad_(True).to(device)
            # real = real.view(-1, x_size * y_size).to(device)
            b_size = real.shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            # output = netD(real_cpu).view(-1)
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))

            real_output = netD(real).view(-1)
            if shape_stat == 1:
                print(real_output.shape)
                print("Out:NetD - real batch\n--------------------------\n")
                time.sleep(2)
            # Calculate loss on all-real batch
            errD_real = criterion(real_output, label)
            # Calculate gradients for D in backward pass
            # errD_real.backward()
            # D_x = output.mean().item()
            if netD.training == 1:
                errD_real.backward()
            D_x = real_output.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors

            # noise = torch.reshape(torch.squeeze(torch.randn(b_size, nz, 1)), (b_size, nz, 1, 1)).to(device)
            # noise = torch.reshape((noise,(100,50,1,1)))
            noise = ((r_min - r_max) * torch.reshape(torch.squeeze(torch.randn(b_size, nz, 1)),
                                                     (b_size, nz, 1, 1)) + r_max).to(device)
            # noise = torch.rand(b_size, nz, nz, device=device)
            if shape_stat == 1:
                print(noise.shape)
                print("Random noise\n--------------------------\n")
                time.sleep(2)
            # Generate fake image batch with G
            # fake = torch.reshape(torch.squeeze(F.interpolate(netG(noise), size=grid_size)),
            #                      (b_size, 1, grid_size, grid_size)).to(device)
            fake = netG(noise).to(device)
            if shape_stat == 1:
                print(fake.shape)
                print("NetG out - fake gen\n--------------------------\n")
                time.sleep(2)
            # print(fake.shape)
            # time.sleep(4)
            label.fill_(fake_label)

            # Classify all fake batch with D
            fake_output = netD(fake.detach()).view(-1)
            if shape_stat == 1:
                print(fake_output.shape)
                print("NetD out - fake disc\n--------------------------\n")
                time.sleep(2)
            # output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(fake_output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            if netD.training == 1:
                errD_fake.backward()
                # Compute error of D as sum over the fake and the real batches
                # Update D
                optimizerD.step()
            errD = errD_real + errD_fake
            D_G_z1 = fake_output.mean().item()
            # for p in netD.parameters():
            #     p.data.clamp_(-clip, clip)
            if epoch % 2 == 0:
                # Clipping Discriminator Weight
                for p in netD.parameters():
                    p.data.clamp_(-clip, clip)
            else:
                pass

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # netD.zero_grad(set_to_none=True)
            netG.zero_grad(set_to_none=True)
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            fake_output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(fake_output, label)
            # Calculate gradients for G
            if netG.training == 1:
                errG.backward()
                # Update G
                optimizerG.step()

            D_G_z2 = fake_output.mean().item()

            # Output training stats
            if batch_idx % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, batch_idx, len(dataloader_structures),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 10 == 0) or ((epoch == num_epochs - 1) and (batch_idx == len(dataloader_structures) - 1)):
                with torch.no_grad():
                    fake_interpol_img = F.interpolate(netG(fixed_noise), size=grid_size)
                    faket = torch.reshape(torch.squeeze(fake_interpol_img),
                                          (b_size, grid_size, grid_size)).cpu().detach().numpy()
                    ccmap = cm.PRGn
                    fake_learning0 = azes[0].imshow(faket[0], cmap=ccmap)
                    fake_learning1 = azes[1].imshow(faket[1], cmap=ccmap)
                    fake_learning2 = azes[2].imshow(faket[2], cmap=ccmap)
                    fake_learning3 = azes[3].imshow(faket[3], cmap=ccmap)
                    fake_learning4 = azes[4].imshow(faket[4], cmap=ccmap)
                    fake_learning5 = azes[5].imshow(faket[5], cmap=ccmap)
                    fake_learning6 = azes[6].imshow(faket[6], cmap=ccmap)
                    fake_learning7 = azes[7].imshow(faket[7], cmap=ccmap)
                    fake_learning8 = azes[8].imshow(faket[8], cmap=ccmap)
                    fake_learning9 = azes[9].imshow(faket[9], cmap=ccmap)
                    loss_G, = azes[10].plot(G_losses, color="blue")
                    loss_D, = azes[10].plot(D_losses, color="red")
                    azes[10].set_title("Generator and Discriminator Loss During Training")
                    azes[10].set_xlabel("iterations")
                    azes[10].set_ylabel("Loss")
                    if epoch == num_epochs - 1:
                        if y_top == 0 and max(G_losses) > max(D_losses):
                            y_top = max(G_losses)
                        elif y_top == 0 and max(G_losses) < max(D_losses):
                            y_top = max(D_losses)
                        else:
                            pass
                        azes[10].text(
                            num_epochs * len(dataloader_structures) + num_epochs * len(dataloader_structures) / 15,
                            y_top - y_top / 2, 'G', color="blue", fontsize=15.)
                        azes[10].text(
                            num_epochs * len(dataloader_structures) + num_epochs * len(dataloader_structures) / 15,
                            y_top - y_top / 4, 'D', color="red",
                            fontsize=15.)
                    realt = torch.reshape(real, (b_size, grid_size, grid_size)).cpu().detach().numpy()
                    modesF = azes[11].scatter(faket[:, 0],
                                              faket[:, 1],
                                              edgecolor='red', facecolor='None', s=5, alpha=1,
                                              linewidth=1, label='GAN')
                    modesR = azes[11].scatter(realt[:, 0],
                                              realt[:, 1],
                                              edgecolor='blue', facecolor='None', s=5, alpha=1,
                                              linewidth=1, label='Real')
                    azes[11].set_title("MODES | Real - blue | Fake - Red")
                    test_data_index = 0
                    for i in range(batch_size):
                        test_tensor = torch.tensor(test_data[i]).to(device)
                        test_input = torch.unsqueeze(test_tensor.float(), 1)
                        ## ([batch_size, 1, 250, 250])
                        c = torch.squeeze(netD(test_input))
                        confidences[i] = c

                    conf = azes[12].imshow(confidences.cpu().detach().numpy(), cmap=cm.bwr)
                    img.append(
                        [fake_learning0, fake_learning1, fake_learning2, fake_learning3, fake_learning4, fake_learning5,
                         fake_learning6, fake_learning7, fake_learning8, fake_learning9, loss_G, loss_D, modesF,
                         modesR, conf])
            iters += 1
    if disp_progrss == 1 or lap_counter == laps - 1:
        ani = animation.ArtistAnimation(fakes_loss_modes, img, interval=30, blit=True)
        plt.show()
        #ani.save('../resGAN6.gif', writer='pillow', fps=25, dpi=100)

    else:
        pass
    ############## Save Model After Training ###############
    torch.save(netD.state_dict(), pathD)
    torch.save(netG.state_dict(), pathG)
