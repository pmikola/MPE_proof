import FDTD_2d
from C import C
import numpy as np
from matplotlib import pyplot as plt
import sys
import matplotlib.cm as cm
from numba import cuda, vectorize, guvectorize, jit, njit
# ------------------------------- INIT --------------------------
cc = C()
plot_flag = 1
show_structure = 1
particle_scale = 2  # 0 = lambda, 1 = lambda/10, else = lambda*10
num_of_structures = 1
grid_size = 250
pulse_len = 100
freq = 454.231E12
nsteps = 100
pulse_loc_x = round(grid_size / 2)
pulse_loc_y = round(grid_size / 2)  # 3
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


# ------------------------------- FUNCTIONS --------------------------

# @jit(nopython=True, parallel=True)
# def GenStructure(x,y,grid_size):
#     struct = np.zeros(grid_size, dtype=np.float64)
#     for i in range(0, len(x)):
#         for j in range(0, len(y)):
#             struct[x[i]] = y[j]
#     Generated_Structure = [np.array(range(0, grid_size)), struct]
#     return Generated_Structure

# ------------------------------- FUNCTIONS --------------------------

# ------------------------------- DATA --------------------------
# INPUT DATA FOR GAN NETWORK
np.set_printoptions(threshold=sys.maxsize)
x = np.array(SIM.x_points)
y = np.array(SIM.y_points)
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
ay = fig.add_subplot(grid[:, :])
x_fig = range(0, grid_size)
y_fig = range(0, grid_size)
ay.plot(x_fig, y_fig, alpha=0)
# ay.scatter(x, y, c='grey', s=70, alpha=0.017)
Generated_Structure = np.zeros((2,1),dtype=np.float32)
pairs = np.zeros((2,1),dtype=np.float32)
#print(Generated_Structure)
for i in range(0,len(x)):
    pairs[0] = x[i]
    pairs[1] = y[i]
    #print(pairs)
    Generated_Structure = np.append(Generated_Structure, pairs,axis=1)
#     #for j in range(0, len(y)):
#     #np.append(Generated_Structure,)
    #print(Generated_Structure)
#
Generated_Structure = np.unique(Generated_Structure,axis=0)
Generated_Structure = np.delete(Generated_Structure,0,axis=1)
Gen_Structure = np.zeros((grid_size,grid_size),dtype=np.float32)
#print(Generated_Structure[0][:])
for i in range(0,len(Generated_Structure[0])-1):
    x_pair = Generated_Structure[0][i]
    y_pair = Generated_Structure[1][i]
    Gen_Structure[int(x_pair)][int(y_pair)] = 1.


#print(Gen_Structure)
# #print(Struct)
ay.imshow(Gen_Structure,origin='upper')
#ay.scatter(Generated_Structure[0])
plt.grid()
plt.show()
