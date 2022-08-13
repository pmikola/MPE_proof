# INPUT DATA FOR GAN NETWORK GENERATOR
import os
from os import walk
import FDTD_2d
from C import C
import numpy as np
from matplotlib import pyplot as plt, animation
import sys
import matplotlib.cm as cm
from numba import cuda, vectorize, guvectorize, jit, njit
import time
import random as _random
import pickle


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
        if x_pair > grid_size - 1 or y_pair > grid_size - 1:
            #print(x_pair, y_pair)
            pass
        else:
            Gen_Structure[int(x_pair)][int(y_pair)] = 1.
    #Gen_Structure[pulse_loc_x][pulse_loc_y] = 2. # for later purpose
    #     # print([int(x_pair), int(y_pair)])
    #     # print(Gen_Structure[int(x_pair)][int(y_pair)])
    #     # print(Gen_Structure[int(x_pair)][:])
    Gen_Structure = np.rot90(Gen_Structure, 1, axes=(0, 1))
    return SIM, Gen_Structure


def findSubstringInFiles(f, names_struct, names_field, names_meta, save_flag,
                         path):
    if save_flag == 0:
        substring1 = "Structure"
        substring2 = "Field"
        substring3 = "Meta_Data"
        for (dirpath, dirnames, filenames) in walk(path):
            f.extend(filenames)
            break
        if len(f) == 0:
            pass
        else:
            for i in range(0, len(f)):
                if f[i].find(substring1) != -1:
                    names_struct.append(f[i])
                else:
                    pass
            for i in range(0, len(f)):
                if f[i].find(substring2) != -1:
                    names_field.append(f[i])
                else:
                    pass
            for i in range(0, len(f)):
                if f[i].find(substring3) != -1:
                    names_meta.append(f[i])
                else:
                    pass
    return f, names_field, names_struct, names_meta


def Loader(i, names_struct, names_field, names_meta, path):
    Loaded_Structure = np.load(os.path.join(path, names_struct[i]))
    Loaded_Field = np.load(os.path.join(path, names_field[i]))
    Loaded_Meta = np.load(os.path.join(path, names_meta[i]))
    return Loaded_Structure, Loaded_Field, Loaded_Meta


def Check_Data(check_data, check_file_dataset, nsteps, frame_interval, grid_size, names_struct, names_field, names_meta,
               path, save_flag):
    if check_data == 1:

        T = 0
        ims = []
        print("SavedDataVisCheck")
        f = []
        # print(f)
        f, names_field, names_struct, names_meta = findSubstringInFiles(f,
                                                                        names_struct,
                                                                        names_field, names_meta, save_flag, path)
        Loaded_Structure = np.load(os.path.join(path, names_struct[check_file_dataset]))
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
        ay = fig.add_subplot(grid[:, :])
        ay.imshow(Loaded_Structure)
        plt.show()
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
        ay = fig.add_subplot(grid[:, :])
        Loaded_Field = np.load(os.path.join(path, names_field[check_file_dataset]))
        Loaded_Meta = np.load(os.path.join(path, names_meta[check_file_dataset]))
        ddx = Loaded_Meta[5]
        for n in range(0, nsteps):
            T += 1
            if T % frame_interval == 0:
                Z = Loaded_Field[:, grid_size * n - grid_size: grid_size * n]
                ims0 = ay.imshow(Z, cmap=cm.hot,  # interpolation='nearest',
                                 extent=[0, grid_size * ddx, 0, grid_size * ddx])
                ims0.set_interpolation('bilinear')
                ims.append([ims0])
        ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
        plt.show()

        print(Loaded_Meta)
    else:
        pass


def Generate(DataNum, frame_interval, plot_period, grid_size, plot_flag, show_structure, save_flag, check_data, nsteps,
             names_struct, names_field, names_meta, path, check_file_dataset, generate):
    for i in range(0, DataNum):
        if generate == 1:
            num_of_structures = 0
            particle_scale = np.random.uniform(0,2)  # 0 = lambda, 1 = lambda/10, else = lambda*10
            if particle_scale == 0:
                num_of_structures = np.random.randint(1, 20)
            elif particle_scale == 1:
                num_of_structures = np.random.randint(1, 100)
            else:
                num_of_structures = np.random.randint(1, 4)
            pulse_len = np.random.randint(0, grid_size)
            n_index = np.random.uniform(1, 10)  # cc.nSiO2
            sigma = np.random.uniform(1, 10)  # cc.sigmaSiO2
            wavelength = np.random.randint(250, 2500)
            freq = C.WavelengthToFrequency(np.random.uniform(250E-9, 2500E-9), 1)  # 454.231E12
            # time.sleep(10)

            pulse_width = 1
            pulse_height = 1
            pulse_loc_x = np.random.randint(3, grid_size - 3)
            pulse_loc_y = np.random.randint(3, grid_size - 3)

            SIM, Gen_Structure = DataGenerator(frame_interval, plot_period, particle_scale, num_of_structures,
                                               grid_size,
                                               plot_flag,
                                               show_structure,
                                               pulse_len,
                                               freq, nsteps,
                                               pulse_loc_x, pulse_loc_y, pulse_width, pulse_height, n_index, sigma)

            Meta_Data = np.array(
                [nsteps, grid_size, pulse_len, n_index, sigma, wavelength, freq, pulse_width, pulse_height, pulse_loc_x,
                 pulse_loc_y, SIM.dx, SIM.ddx, SIM.dt])
            if save_flag == 1:
                names_field.append("Gen_Field_" + str(i))
                names_struct.append("Gen_Structure_" + str(i))
                names_meta.append("Meta_Data_" + str(i))
                np.save(os.path.join(path, names_field[i]), SIM.FieldProp)
                np.save(os.path.join(path, names_struct[i]), Gen_Structure)
                np.save(os.path.join(path, names_meta[i]), Meta_Data)
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
        else:
            pass
        Check_Data(check_data, check_file_dataset, nsteps, frame_interval, grid_size, names_struct,
                   names_field, names_meta,
                   path, save_flag)

# ------------------------------- FUNCTIONS --------------------------
