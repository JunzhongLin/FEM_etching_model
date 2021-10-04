import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import pandas as pd
from scipy.interpolate import griddata
import pickle


def initiate_mesh(size):  # size means number of steps in spatial dimension

    x = np.arange(-size/2, size/2+1, 1) * 8   # single step is 50 nm
    y = np.arange(-size/2, size/2+1, 1) * 8
    xm, ym = np.meshgrid(x, y)
    zm = np.zeros([size+1, size+1])

    return xm, ym, zm


def ini_round_defect(xm, ym, zm, wx_0, wy_0, d0):  # w0 means the ratio in the sphere equation
    for i in range(len(xm)):
        for j in range(len(ym)):
            if d0**2-wx_0*xm[i][i]**2-wy_0*ym[j][j]**2 < 0:
                zm[i][j] = 0
            else:
                zm[i][j] = -np.sqrt(d0**2-wx_0*xm[i][i]**2-wy_0*ym[j][j]**2)
    return zm


def surface_reduction(xm, ym, zm, rate, t_step):  # rate = 2000/60, nm/s
    xm_t = np.zeros([len(xm), len(xm)])
    ym_t = np.zeros([len(ym), len(ym)])
    zm_t = np.zeros([len(zm), len(zm)])

    AB = np.stack((xm[:-2, 1:-1]-xm[1:-1, 1:-1], ym[:-2, 1:-1]-ym[1:-1, 1:-1], zm[:-2, 1:-1]-zm[1:-1, 1:-1]), axis=2)
    AC = np.stack((xm[1:-1, 2:]-xm[1:-1, 1:-1], ym[1:-1, 2:]-ym[1:-1, 1:-1], zm[1:-1, 2:]-zm[1:-1, 1:-1]), axis=2)
    AD = np.stack((xm[2:, 1:-1]-xm[1:-1, 1:-1], ym[2:, 1:-1]-ym[1:-1, 1:-1], zm[2:, 1:-1]-zm[1:-1, 1:-1]), axis=2)
    AE = np.stack((xm[1:-1, :-2]-xm[1:-1, 1:-1], ym[1:-1, :-2]-ym[1:-1, 1:-1], zm[1:-1, :-2]-zm[1:-1, 1:-1]), axis=2)
    n1 = np.cross(AC, AB) / np.linalg.norm(np.cross(AC, AB), axis=2).reshape(len(xm)-2, len(ym)-2, 1)
    n2 = np.cross(AE, AB) / np.linalg.norm(np.cross(AE, AB), axis=2).reshape(len(xm)-2, len(ym)-2, 1)
    n3 = np.cross(AD, AE) / np.linalg.norm(np.cross(AD, AE), axis=2).reshape(len(xm)-2, len(ym)-2, 1)
    n4 = np.cross(AC, AD) / np.linalg.norm(np.cross(AC, AD), axis=2).reshape(len(xm)-2, len(ym)-2, 1)
    N = (n1 + n2 + n3 + n4) / np.linalg.norm(n1 + n2 + n3 + n4, axis=2).reshape(len(xm)-2, len(ym)-2, 1)
    Q = 0.4 * (1 - np.exp(-(4 * zm[1:-1, 1:-1] - zm[2:, 1:-1] - zm[1:-1, 2:] - zm[1:-1, :-2] - zm[:-2, 1:-1]) ** 2))

    xm_t[1:-1, 1:-1] = xm[1:-1, 1:-1] - rate * t_step * N[:, :, 0] * (1 + Q)
    ym_t[1:-1, 1:-1] = ym[1:-1, 1:-1] - rate * t_step * N[:, :, 1] * (1 + Q)
    zm_t[1:-1, 1:-1] = zm[1:-1, 1:-1] - rate * t_step * N[:, :, 2] * (1 + Q)

    # Boundary will not move
    xm_t[0, :] = xm[0, :]
    xm_t[len(xm_t)-1, :] = xm[len(xm_t)-1, :]
    xm_t[:, 0] = xm[:, 0]
    xm_t[:, len(xm_t)-1] = xm[:, len(xm_t)-1]

    ym_t[0, :] = ym[0, :]
    ym_t[len(ym_t)-1, :] = ym[len(ym_t)-1, :]
    ym_t[:, 0] = ym[:, 0]
    ym_t[:, len(ym_t)-1] = ym[:, len(ym_t)-1]

    zm_t[0, :] = zm_t[1, :]
    zm_t[len(zm_t)-1, :] = zm_t[len(zm_t)-2, :]
    zm_t[:, 0] = zm_t[:, 1]
    zm_t[:, len(zm_t)-1] = zm_t[:, len(zm_t)-2]

    return xm_t, ym_t, zm_t, N, Q     # _t means after one time step


# def interpolation_mesh(xm_t, ym_t, zm_t, xm, ym, zm):
#     zm = griddata(np.stack((xm_t.flatten(), ym_t.flatten()), axis=1), zm_t.flatten(),
#                   (xm, ym), method='linear')
#     return zm


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def interpolation_mesh(xm_t, ym_t, zm_t, xm, ym, zm):
    zm = zm_t
    for i in np.arange(1, len(xm)-1, 1):
        for j in np.arange(1, len(ym)-1, 1):
            if xm[i, j] != xm_t[i, j] or ym[i, j] != ym_t[i, j]:
                # print(i, j)
                for a in [-1, 1]:
                    for b in [-1, 1]:
                        s1 = area(xm[i, j], ym[i, j], xm_t[i+a, j], ym_t[i+a, j], xm_t[i, j+b], ym_t[i, j+b])
                        s2 = area(xm[i, j], ym[i, j], xm_t[i, j], ym_t[i, j], xm_t[i, j+b], ym_t[i, j+b])
                        s3 = area(xm[i, j], ym[i, j], xm_t[i+a, j], ym_t[i+a, j], xm_t[i, j], ym_t[i, j])
                        s_total = area(xm_t[i+a, j], ym_t[i+a, j], xm_t[i, j], ym_t[i, j], xm_t[i, j+b], ym_t[i, j+b])
                        if int(s_total) == int(s1+s2+s3):
                            # print('found the triangle', i, j, a, b)
                            AB = [xm_t[i+a, j]-xm_t[i, j], ym_t[i+a, j]-ym_t[i, j], zm[i+a, j]-zm[i, j]]
                            AC = [xm_t[i, j+b]-xm_t[i, j], ym_t[i, j+b]-ym_t[i, j], zm[i, j+b]-zm[i, j]]
                            nn = np.cross(AC, AB)
                            zm[i, j] = zm_t[i, j] - nn[0]/nn[2]*(xm[i, j]-xm_t[i, j])-nn[1]/nn[2]*(ym[i, j]-ym_t[i, j])
                            break
    return zm


def plot_mesh(xm, ym, zm, size, title):

    plt.rcParams['savefig.dpi'] = 1200
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xm, ym, zm, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax.set_ylabel('width_y (nm)', ) # fontsize=13
    ax.set_xlabel('width_x (nm)', )
    ax.set_zlabel('depth_z (nm)', )
    ax.set_xlim(-size * 8 / 2, size * 8 / 2)
    ax.set_ylim(-size * 8 / 2, size * 8 / 2)
    ax.set_zlim(-size * 8, 0)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_tick_params()  # labelsize=12
    ax.yaxis.set_tick_params()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.grid(True)

    return ax


size = 1200
xm, ym, zm = initiate_mesh(size)
zm = ini_round_defect(xm, ym, zm, 64, 1, 1000)
plot_mesh(xm, ym, zm, size, 'before etching')
for i in range(1200):
    print(i)
    xm_t, ym_t, zm_t, N, Q = surface_reduction(xm, ym, zm, 2000/60, 0.05)
    zm = interpolation_mesh(xm_t, ym_t, zm_t, xm, ym, zm)
plot_mesh(xm, ym, zm, size, 'after etching')


def main():
    pass


def quick_check(i, j, xm_t, ym_t, xm, ym):

    fig, ax = plt.subplots(1, 1)
    ax.scatter(xm_t[i, j], ym_t[i, j], label='O')
    ax.scatter(xm_t[i-1, j], ym_t[i-1, j], label='A')
    ax.scatter(xm_t[i, j+1], ym_t[i, j+1], label='B')
    ax.scatter(xm_t[i+1, j], ym_t[i+1, j], label='C')
    ax.scatter(xm_t[i, j-1], ym_t[i, j-1], label='D')
    ax.scatter(xm[i, j], ym[i, j], label='origin')
    ax.legend()

    return ax