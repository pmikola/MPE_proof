import FDTD_2d
from C import C
import numpy as np
from matplotlib import pyplot as plt
import sys
# ------------------------------- INIT --------------------------
cc = C()
plot_flag = 0
show_structure = 0
grid_size = 500
pulse_len = 200
freq = 454.231E12
nsteps = 10
pulse_loc_x = round(grid_size / 2)
pulse_loc_y = 3
particle_scale = 0  # 0 = lambda, 1 = lambda/10, else = lambda*10
num_of_structures = 5
n_index = cc.nSiO2
sigma = cc.sigmaSiO2
# ------------------------------- INIT --------------------------

SIM = FDTD_2d.FDTD(particle_scale, num_of_structures, grid_size, plot_flag, show_structure, pulse_len, freq, nsteps,
                   pulse_loc_x, pulse_loc_y, n_index, sigma)
SIM.PML()
SIM.ShapeGen()
SIM.medium()
SIM.CORE()
SIM.plot_sim()

# ------------------------------- DATA --------------------------
# INPUT DATA FOR GAN NETWORK
np.set_printoptions(threshold=sys.maxsize)
x = np.array(SIM.x_points)
y = np.array(SIM.y_points)
print(x)
print(y)
print(x.shape)
print(y.shape)

Struct = np.zeros((grid_size, grid_size), dtype=np.float32)

# x_unique = np.unique(x)
# y_unique = np.unique(y)
#
# print(x_unique)
# print(y_unique)
# print(x_unique.shape)
# print(y_unique.shape)


fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
ay = fig.add_subplot(grid[:, :])
ay.scatter(np.array(SIM.x_points), np.array(SIM.y_points), c='grey', s=70, alpha=0.015)
plt.show()
