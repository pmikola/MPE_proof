# INPUT DATA FOR GAN NETWORK GENERATOR
import os

import FDTD_2d
from C import C
import numpy as np
from matplotlib import pyplot as plt, animation
import sys
import matplotlib.cm as cm
from numba import cuda, vectorize, guvectorize, jit, njit
import time
import pickle

# ------------------------------- INIT --------------------------
cc = C()
path = 'DATASET/training'
names_field = []
names_struct = []
names_meta = []
plot_flag = 0
show_structure = 0
plot_period = 5000  # ms
grid_size = 250  # 300
nsteps = 500  # 750
save_flag = 0
check_data = 0
DataNum = 1
frame_interval = 8




# np.random.seed(2022)
# np.random.seed(828)

# ------------------------------- INIT --------------------------
# ------------------------------- FUNCTIONS --------------------------
def DataGenerator(frame_interval, plot_period, particle_scale, num_of_structures, grid_size, plot_flag, show_structure,
                  pulse_len, freq,
                  nsteps,
                  pulse_loc_x, pulse_loc_y, pulse_width, pulse_height, n_index, sigma):
    SIM = FDTD_2d.FDTD(frame_interval, plot_period, particle_scale, num_of_structures, grid_size, plot_flag,
                       show_structure, pulse_len,
                       freq, nsteps,
                       pulse_loc_x, pulse_loc_y, pulse_width, pulse_height, n_index, sigma)
    SIM.PML()
    SIM.ShapeGen()
    SIM.medium()
    SIM.CORE()
    SIM.plot_sim()

    np.set_printoptions(threshold=sys.maxsize)
    x = np.array(SIM.x_points)
    y = np.array(SIM.y_points)
    # x_fig = range(0, grid_size)
    # y_fig = range(0, grid_size)

    # ay.scatter(x, y, c='grey', s=70, alpha=0.017)
    Generated_Structure = np.zeros((2, 1), dtype=np.float32)
    pairs = np.zeros((2, 1), dtype=np.float32)
    # print(Generated_Structure)
    for i in range(0, len(x)):
        pairs[0] = x[i]
        pairs[1] = y[i]
        # print(pairs)
        Generated_Structure = np.append(Generated_Structure, pairs, axis=1)
    #     #for j in range(0, len(y)):
    #     #np.append(Generated_Structure,)
    # print(Generated_Structure)
    #
    Generated_Structure = np.unique(Generated_Structure, axis=1)
    Generated_Structure = np.delete(Generated_Structure, 0, axis=1)
    Gen_Structure = np.zeros((grid_size, grid_size), dtype=np.float32)
    # print(Generated_Structure[0][:])

    # print(Generated_Structure[0].shape)
    # print(Generated_Structure[1].shape)
    for i in range(0, len(Generated_Structure[0])):
        x_pair = Generated_Structure[0][i]
        y_pair = Generated_Structure[1][i]
        if x_pair > grid_size or y_pair > grid_size:
            pass
        else:
            Gen_Structure[int(x_pair)][int(y_pair)] = 1.
    Gen_Structure[pulse_loc_x][pulse_loc_y] = 2.
    #     # print([int(x_pair), int(y_pair)])
    #     # print(Gen_Structure[int(x_pair)][int(y_pair)])
    #     # print(Gen_Structure[int(x_pair)][:])
    Gen_Structure = np.rot90(Gen_Structure, 1, axes=(0, 1))
    return SIM, Gen_Structure


def Check_Data(SIM, check_data, nsteps, frame_interval, grid_size, names_struct):
    if check_data == 1:
        T = 0
        ims = []
        print("SavedDataVisCheck")
        Loaded_Structure = np.load(os.path.join(path, names_struct[0] + '.npy'))
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
        ay = fig.add_subplot(grid[:, :])
        ay.imshow(Loaded_Structure)
        plt.show()
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
        ay = fig.add_subplot(grid[:, :])
        Loaded_Field = np.load(os.path.join(path, names_field[0] + '.npy'))
        for n in range(0, nsteps):
            T += 1
            if T % frame_interval == 0:
                Z = Loaded_Field[:, grid_size * n - grid_size: grid_size * n]
                ims0 = ay.imshow(Z, cmap=cm.hot,  # interpolation='nearest',
                                 extent=[0, grid_size * SIM.ddx, 0, grid_size * SIM.ddx])
                ims0.set_interpolation('bilinear')
                ims.append([ims0])
        ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
        plt.show()
        Loaded_Meta = np.load(os.path.join(path, names_meta[0] + '.npy'))
        print(Loaded_Meta)
    else:
        pass


# ------------------------------- FUNCTIONS --------------------------

# ------------------------------- DATA GEN --------------------------
for i in range(0, DataNum):
    particle_scale = 0  # 0 = lambda, 1 = lambda/10, else = lambda*10
    num_of_structures = np.random.randint(1, 20)
    pulse_len = np.random.randint(0, grid_size)
    n_index = np.random.uniform(1, 10)  # cc.nSiO2
    sigma = np.random.uniform(1, 10)  # cc.sigmaSiO2
    wavelength = np.random.randint(250, 2500)
    freq = C.WavelengthToFrequency(660E-9, 1)  # 454.231E12
    # time.sleep(10)

    pulse_width = 1
    pulse_height = 1
    pulse_loc_x = np.random.randint(3, grid_size - 3)
    pulse_loc_y = np.random.randint(3, grid_size - 3)

    SIM, Gen_Structure = DataGenerator(frame_interval, plot_period, particle_scale, num_of_structures, grid_size,
                                       plot_flag,
                                       show_structure,
                                       pulse_len,
                                       freq, nsteps,
                                       pulse_loc_x, pulse_loc_y, pulse_width, pulse_height, n_index, sigma)

    Meta_Data = np.array(
        [nsteps, grid_size, pulse_len, n_index, sigma, wavelength, freq, pulse_width, pulse_height, pulse_loc_x,
         pulse_loc_y])
    if save_flag == 1:
        names_field.append("Gen_Field_" + str(i))
        names_struct = ["Gen_Structure_" + str(i)]
        names_meta = ["Meta_Data_" + str(i)]
        np.save(os.path.join(path, names_field[i]), SIM.FieldProp)
        np.save(os.path.join(path, names_struct[i]), Gen_Structure)
        np.save(os.path.join(path, names_meta[i]), Meta_Data)

    else:
        pass

    if plot_flag == 1:
        if show_structure == 1:
            fig = plt.figure(figsize=(5, 5))
            timer = fig.canvas.new_timer(interval=5000)
            timer.add_callback(plt.close)
            grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
            ay = fig.add_subplot(grid[:, :])
            # ay.plot(x_fig, y_fig, alpha=0)
            ay.imshow(Gen_Structure)
            timer.start()
            plt.show()
    Check_Data(SIM, check_data, nsteps, frame_interval, grid_size, names_struct)
    del SIM, Gen_Structure
