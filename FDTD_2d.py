import math as M
import time
from numba import cuda, vectorize, guvectorize, jit, njit
import numpy as np
from matplotlib import pyplot as plt, animation, offsetbox
import matplotlib.cm as cm
from C import C
import cupy as cp
import cairo


class FDTD_2D:
    def __init__(self):
        # mpl.use('Agg')
        self.LetsPlot = 1
        jit(device=True)
        jit(warn=False)
        np.seterr(divide='ignore', invalid='ignore')
        self.s = time.time()
        self.nett_time_sum = 0
        self.frame_interval = 16
        self.ims = []
        self.flag = 1
        self.data_type1 = np.float32
        self.cc = C()
        self.IE = self.JE = 500  # GRID
        self.npml = 8
        self.freq = FDTD_2D.data_type(self, 454.231E12)  # red light (666nm)
        self.n_index = self.cc.nSiO2
        self.n_sigma = self.cc.sigmaSiO2
        self.epsilon = FDTD_2D.data_type(self, self.n_index)
        self.sigma = FDTD_2D.data_type(self, self.n_sigma)
        self.epsilon_medium = FDTD_2D.data_type(self, 1.003)
        self.sigma_medium = FDTD_2D.data_type(self, 0.)
        self.wavelength = self.cc.c0 / (self.n_index * self.freq)
        self.vm = self.wavelength * self.freq
        self.dx = 10
        self.ddx = FDTD_2D.data_type(self, self.wavelength / self.dx)  # Cells Size
        # dt = data_type((ddx / cc.c0) *  M.sqrt(2),flag) # Time step
        #   CFL stability condition- Lax Equivalence Theorem
        self.dt = 1 / (self.vm * M.sqrt(1 / (self.ddx ** 2) + 1 / (self.ddx ** 2)))  # Tiem step\
        self.z_max = FDTD_2D.data_type(self, 0)
        self.epsz = FDTD_2D.data_type(self, 8.854E-12)
        self.spread = FDTD_2D.data_type(self, 8)
        self.t0 = FDTD_2D.data_type(self, 1)
        self.ic = self.IE / 2
        self.jc = self.JE / 2
        self.ia = 7  # total scattered field boundaries
        self.ib = self.IE - self.ia - 1
        self.ja = 7
        self.jb = self.JE - self.ja - 1
        self.nsteps = 1500
        self.T = 0
        self.medium_eps = 1. / (self.epsilon_medium + self.sigma_medium * self.dt / self.epsz)
        self.medium_sigma = self.sigma_medium * self.dt / self.epsz
        self.INTEGRATE = []
        self.x_points = []
        self.y_points = []
        self.window = 10
        self.k_vec = 2 * M.pi / self.wavelength
        self.omega = 2 * M.pi * self.freq

        self.ez_inc_low_m2 = FDTD_2D.data_type(self, 0.)
        self.ez_inc_low_m1 = FDTD_2D.data_type(self, 0.)

        self.ez_inc_high_m2 = FDTD_2D.data_type(self, 0.)
        self.ez_inc_high_m1 = FDTD_2D.data_type(self, 0.)

        self.dz = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.iz = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.ez = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.hx = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.hy = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.ihx = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.ihy = np.zeros((self.IE, self.JE), dtype=self.data_type1)
        self.ga = np.ones((self.IE, self.JE), dtype=self.data_type1) * self.medium_eps  # main medium epsilon
        self.gb = np.zeros((self.IE, self.JE), dtype=self.data_type1) * self.medium_sigma  # main medium sigma
        self.Pz = np.zeros((self.IE, self.JE), dtype=self.data_type1)

        self.gi2 = np.ones(self.IE, dtype=self.data_type1)
        self.gi3 = np.ones(self.IE, dtype=self.data_type1)
        self.fi1 = np.zeros(self.IE, dtype=self.data_type1)
        self.fi2 = np.ones(self.IE, dtype=self.data_type1)
        self.fi3 = np.ones(self.IE, dtype=self.data_type1)

        self.gj2 = np.ones(self.JE, dtype=self.data_type1)
        self.gj3 = np.ones(self.JE, dtype=self.data_type1)
        self.fj1 = np.zeros(self.JE, dtype=self.data_type1)
        self.fj2 = np.ones(self.JE, dtype=self.data_type1)
        self.fj3 = np.ones(self.JE, dtype=self.data_type1)

        self.ez_inc = np.zeros(self.JE, dtype=self.data_type1)
        self.hx_inc = np.zeros(self.JE, dtype=self.data_type1)
        self.netend = 0.
        self.nett_time_sum = 0.
        self.pulse = 0.
        self.net = 0.
        self.data = np.zeros((self.IE, self.JE, 4), dtype=np.uint8)
        self.shape1 = self.data[:, :, 0].shape[0]
        self.shape2 = self.data[:, :, 0].shape[1]
        self.fig = plt.figure(figsize=(5, 5))
        self.grid = plt.GridSpec(20, 20, wspace=10, hspace=0.6)
        self.ay = self.fig.add_subplot(self.grid[:, :])
        # Cyclic Number of image snapping
        x = np.linspace(0, self.JE, self.JE)
        y = np.linspace(0, self.IE, self.IE)
        # values = range(len(x))
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = None

    # -------------------------------- KERNELS ---------------------------
    @staticmethod
    @jit(nopython=True, parallel=True)
    def Ez_inc_CU(JE, ez_inc, hx_inc):
        for j in range(1, JE):
            ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
        return ez_inc

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Dz_CU(IE, JE, dz, hx, hy, gi2, gi3, gj2, gj3):
        for j in range(1, JE):
            for i in range(1, IE):
                dz[i][j] = gi3[i] * gj3[j] * dz[i][j] + \
                           gi2[i] * gj2[j] * 0.5 * \
                           (hy[i][j] - hy[i - 1][j] -
                            hx[i][j] + hx[i][j - 1])
        return dz

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Dz_inc_val_CU(ia, ib, ja, jb, dz, hx_inc):
        for i in range(ia, ib + 1):
            dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja - 1]
            dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb]
        return dz

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Ez_Dz_CU(IE, JE, ez, ga, gb, dz, iz):
        for j in range(0, JE):
            for i in range(0, IE):
                ez[i, j] = ga[i, j] * (dz[i, j] - iz[i, j])
                iz[i, j] = iz[i, j] + gb[i, j] * ez[i, j]
        return ez, iz

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Hx_inc_CU(JE, hx_inc, ez_inc):
        for j in range(0, JE - 1):
            hx_inc[j] = hx_inc[j] + .5 * (ez_inc[j] - ez_inc[j + 1])
        return hx_inc

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Hx_CU(IE, JE, ez, hx, ihx, fj3, fj2, fi1):
        for j in range(0, JE - 1):
            for i in range(0, IE - 1):
                curl_e = ez[i][j] - ez[i][j + 1]
                ihx[i][j] = ihx[i][j] + curl_e
                hx[i][j] = fj3[j] * hx[i][j] + fj2[j] * \
                           (.5 * curl_e + fi1[i] * ihx[i][j])
        return ihx, hx

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Hx_inc_val_CU(ia, ib, ja, jb, hx, ez_inc):
        for i in range(ia, ib + 1):
            hx[i][ja - 1] = hx[i][ja - 1] + .5 * ez_inc[ja]
            hx[i][jb] = hx[i][jb] - .5 * ez_inc[jb]
        return hx

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Hy_CU(IE, JE, hy, ez, ihy, fi3, fi2, fi1):
        for j in range(0, JE):
            for i in range(0, IE - 1):
                curl_e = ez[i][j] - ez[i + 1][j]
                ihy[i][j] = ihy[i][j] + curl_e
                hy[i][j] = fi3[i] * hy[i][j] - fi2[i] * \
                           (.5 * curl_e + fi1[j] * ihy[i][j])
        return ihy, hy

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Hy_inc_CU(ia, ib, ja, jb, hy, ez_inc):
        for j in range(ja, jb + 1):
            hy[ia - 1][j] = hy[ia - 1][j] - .5 * ez_inc[j]
            hy[ib][j] = hy[ib][j] + .5 * ez_inc[j]
        return hy

    @staticmethod
    @jit(nopython=True, parallel=True)
    def Power_Calc(IE, JE, Pz, ez, hy, hx):
        for j in range(0, JE):
            for i in range(0, IE):
                Pz[i][j] = M.sqrt(M.pow(-ez[i][j] * hy[i][j], 2) + M.pow(ez[i][j] * hx[i][j], 2))
        return Pz

    # -------------------------------- KERNELS ---------------------------

    # ------------------------------- FUNCTIONS --------------------------
    def data_type(self, data):
        if self.flag == 1:
            return np.float32(data)
        else:
            return np.float64(data)

    # PML Definition
    def PML(self):
        for i in range(self.npml):
            xnum = self.npml - i
            xd = self.npml
            xxn = xnum / xd
            xn = 0.33333 * pow(xxn, 3)
            self.gi2[i] = 1. / (1. + xn)
            self.gi2[self.IE - 1 - i] = 1. / (1. + xn)
            self.gi3[i] = (1. - xn) / (1. + xn)
            self.gi3[self.IE - i - 1] = (1. - xn) / (1. + xn)
            xxn = (xnum - .5) / xd
            xn = 0.33333 * pow(xxn, 3)
            self.fi1[i] = xn
            self.fi1[self.IE - 2 - i] = xn
            self.fi2[i] = 1.0 / (1.0 + xn)
            self.fi2[self.IE - 2 - i] = 1.0 / (1.0 + xn)
            self.fi3[i] = (1.0 - xn) / (1.0 + xn)
            self.fi3[self.IE - 2 - i] = (1.0 - xn) / (1.0 + xn)

            self.gj2[i] = 1. / (1. + xn)
            self.gj2[self.JE - 1 - i] = 1. / (1. + xn)
            self.gj3[i] = (1.0 - xn) / (1. + xn)
            self.gj3[self.JE - i - 1] = (1. - xn) / (1. + xn)
            xxn = (xnum - .5) / xd
            xn = 0.33333 * pow(xxn, 3)
            self.fj1[i] = xn
            self.fj1[self.JE - 2 - i] = xn
            self.fj2[i] = 1. / (1. + xn)
            self.fj2[self.JE - 2 - i] = 1. / (1. + xn)
            self.fj3[i] = (1. - xn) / (1. + xn)
            self.fj3[self.JE - 2 - i] = (1. - xn) / (1. + xn)
        return self.gi2, self.gi3, self.fi1, self.fi2, self.fi3, self.gj2, self.gj3, self.fj1, self.fj2, self.fj3

    def ShapeGen(self):
        surface = cairo.ImageSurface.create_for_data(
            self.data, cairo.FORMAT_ARGB32, self.IE, self.JE)
        cr = cairo.Context(surface)

        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.paint()
        for i in range(0, int(np.random.randint(1, 2, 1))):
            x1 = np.random.randint(0, self.IE, 1)
            x2 = np.random.randint(0, self.IE, 1)
            y1 = np.random.randint(0, self.JE, 1)
            y2 = np.random.randint(0, self.JE, 1)
            cr.rectangle(x1, x2, y1, y2)
        for i in range(0, int(np.random.randint(1, 2, 1))):
            x1 = np.random.randint(0, self.IE, 1)
            x2 = np.random.randint(0, self.IE, 1)
            y1 = np.random.randint(0, self.JE, 1)
            y2 = np.random.randint(0, self.JE, 1)
            cr.arc(x1, x2, y1, y2, 2 * M.pi)
            cr.set_line_width(5)
            cr.close_path()

        cr.set_source_rgb(1.0, 0.0, 0.0)
        cr.fill()

        return self.data

    def medium(self):
        for j in range(0, self.shape2):
            for i in range(0, self.shape1):
                if self.data[i, j, 0] <= 0:
                    # print(data[i, j, 0])
                    self.ga[j, i] = FDTD_2D.data_type(self, 1 / (self.epsilon + (self.sigma * self.dt / self.epsz)))
                    self.gb[j, i] = FDTD_2D.data_type(self, self.sigma * self.dt / self.epsz)
                    self.x_points.append(i)
                    self.y_points.append(self.JE - j)
                if self.data[i, j, 0] > 0:
                    pass
                    # print(data[i, j, 0])
        return self.ga, self.gb, self.x_points, self.y_points, self.data, self.shape1, self.shape2

    def CORE(self):
        for n in range(1, self.nsteps):
            self.net = time.time()
            self.T = self.T + 1
            # MAIND FDTD LOOP
            # ez_incd, hx_incd = cuda.to_device(ez_inc, stream=stream), cuda.to_device(hx_inc, stream=stream)
            self.ez_inc = FDTD_2D.Ez_inc_CU(self.JE, self.ez_inc, self.hx_inc)
            # ez_inc, hx_inc = ez_incd.copy_to_host(stream=stream), hx_incd.copy_to_host(stream=stream)
            self.ez_inc[0] = self.ez_inc_low_m2
            self.ez_inc_low_m2 = self.ez_inc_low_m1
            self.ez_inc_low_m1 = self.ez_inc[1]
            self.ez_inc[self.JE - 1] = self.ez_inc_high_m2
            self.ez_inc_high_m2 = self.ez_inc_high_m1
            self.ez_inc_high_m1 = self.ez_inc[self.JE - 2]
            self.dz = FDTD_2D.Dz_CU(self.IE, self.JE, self.dz, self.hx, self.hy, self.gi2, self.gi3, self.gj2, self.gj3)
            if self.T < 300:
                self.pulse = FDTD_2D.data_type(self, M.sin(2 * M.pi * self.freq * self.dt * self.T))
                # pulse = data_type(M.exp(-.5 * (pow((t0 - T * 4) / spread, 2))), flag)
                # pulse = data_type(M.exp(-(T-t0)**2/(2*(t0/10)**2)) * M.sin(2*M.pi * (cc.c0/wavelength)*T),flag)
                self.dz[round(self.IE / 2)][3] = self.pulse  # plane wave
            else:
                pass
            self.dz = FDTD_2D.Dz_inc_val_CU(self.ia, self.ib, self.ja, self.jb, self.dz, self.hx_inc)
            self.ez, self.iz = FDTD_2D.Ez_Dz_CU(self.IE, self.JE, self.ez, self.ga, self.gb, self.dz, self.iz)
            self.hx_inc = FDTD_2D.Hx_inc_CU(self.JE, self.hx_inc, self.ez_inc)
            self.ihx, self.hx = FDTD_2D.Hx_CU(self.IE, self.JE, self.ez, self.hx, self.ihx, self.fj3, self.fj2,
                                              self.fi1)
            self.hx = FDTD_2D.Hx_inc_val_CU(self.ia, self.ib, self.ja, self.jb, self.hx, self.ez_inc)
            self.ihy, hy = FDTD_2D.Hy_CU(self.IE, self.JE, self.hy, self.ez, self.ihy, self.fi3, self.fi2, self.fi1)
            self.hy = FDTD_2D.Hy_inc_CU(self.ia, self.ib, self.ja, self.jb, self.hy, self.ez_inc)
            self.Pz = FDTD_2D.Power_Calc(self.IE, self.JE, self.Pz, self.ez, self.hy, self.hx)
            self.netend = time.time()
            # print("Time netto : " + str((netend - net)) + "[s]")
            self.nett_time_sum += self.netend - self.net
            # Drawing of the EM and FT plots
            if self.LetsPlot == 1:
                if self.T % self.frame_interval == 0:
                    self.Z = self.Pz[:][:]  # Power - W/m^2s
                    # self.INTEGRATE.append(self.Z)
                    # self.YY = np.trapz(self.INTEGRATE, axis=0) / self.window
                    # if len(self.INTEGRATE) >= self.window:
                    #     del self.INTEGRATE[0]
                    title = self.ay.annotate("Time :" + '{:<.4e}'.format(self.T * self.dt * 1 * 10 ** 15) + " fs",
                                             (1, 0.5),
                                             xycoords=self.ay.get_window_extent,
                                             xytext=(-round(self.JE * 2), self.IE - 5),
                                             textcoords="offset points", fontsize=9, color='white')
                    ims2 = self.ay.imshow(self.Z, cmap=cm.twilight, extent=[0, self.JE, 0, self.IE])
                    ims2.set_interpolation('bilinear')
                    ims4 = self.ay.scatter(self.x_points, self.y_points, c='blue', s=70, alpha=0.01)
                    self.ims.append([ims2, ims4, title])
                    # print("Punkt : " + str(T))
                else:
                    pass

    def plot_sim(self):
        if self.LetsPlot == 1:
            self.ay.set_xlabel("x [um]")
            self.ay.set_ylabel("y [um]")
            labels = [item.get_text() for item in self.ay.get_xticklabels()]
            labels[0] = '0'
            labels[1] = str(0.2 * self.IE / self.dx)
            labels[2] = str(0.4 * self.IE / self.dx)
            labels[3] = str(0.5 * self.IE / self.dx)
            labels[4] = str(0.8 * self.IE / self.dx)
            labels[5] = str(self.IE / self.dx)
            self.ay.set_xticklabels(labels)
            labels[1] = str(0.2 * self.JE / self.dx)
            labels[2] = str(0.4 * self.JE / self.dx)
            labels[3] = str(0.5 * self.JE / self.dx)
            labels[4] = str(0.8 * self.JE / self.dx)
            labels[5] = str(self.JE / self.dx)
            self.ay.set_yticklabels(labels)

            e = time.time()
            print("Time brutto : " + str((e - self.s)) + "[s]")
            print("Time netto SUM : " + str(self.nett_time_sum) + "[s]")
            file_name = "2d_fdtd_Si_Cylinder_2"
            # file_name = "./" + file_name + '.gif'
            file_name = "./" + file_name + '.gif'
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=30, blit=True)
            # ani.save(file_name, writer='pillow', fps=30, dpi=100)
            # ani.save(file_name + '.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])
            # ani.save(file_name, writer="imagemagick", fps=30)
            print("OK")
            plt.show()
        else:
            pass
    # ------------------------------- FUNCTIONS --------------------------


SIM = FDTD_2D()
SIM.PML()
SIM.ShapeGen()
SIM.medium()
SIM.CORE()
SIM.plot_sim()
