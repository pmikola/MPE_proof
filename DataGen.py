import FDTD_2d

# ------------------------------- INIT --------------------------

plot_flag = 1
show_structure = 0
pulse_len = 200
freq = 454.231E12
nsteps = 2000
# ------------------------------- INIT --------------------------

SIM = FDTD_2d.FDTD(plot_flag, show_structure, pulse_len,freq,nsteps)
SIM.PML()
SIM.ShapeGen()
SIM.medium()
SIM.CORE()
SIM.plot_sim()
