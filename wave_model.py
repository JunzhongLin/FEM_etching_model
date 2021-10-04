import numpy as np
import matplotlib.pyplot as plt


def width_cal(r, rate, time):
    return 2 * rate * time * (np.sqrt((r/2)**2+1)-r/2)


# R = np.arange(0.1, 100, 0.01)
w0 = [0.1, 0.5, 1, 2]
depth0 = np.arange(0.001, 10, 0.1)
R = np.arange(0.01, 40, 0.01)
rate = np.array([2.3/6, 2.3/60, 2.3/600, 2.3/6000])
time_rate = np.array([60, 600, 6000, 60000])
time = np.array([60, 300, 600, 1200])
label_time = ['1min', '5min', '10min', '20min']
label_depth = ['1.5um', '7.5um', '15um', '30um']
label_rate = ['15um/min', '1.5um/min', '0.15um/min', '0.015um/min']
label_w0 = ['w0 = 0.1 um', 'w0 = 0.5 um', 'w0 = 1 um', 'w0 = 2 um']


# for i in range(4):
#     delta_w = width_cal(R, rate[1], time[i])


def delta_width(w0, depth0, rate, time):
    w = 0
    fig, ax = plt.subplots(1, 1)

    for i in range(len(time)):
        w_list = []
        for j in depth0:
            w0_temp = w0[1]
            for k in range(time[i]):
                delta_w = width_cal(w0_temp/j, rate[1], 1)
                w = delta_w + w0_temp
                w0_temp = w
            w_list.append(w)
        ax.plot(depth0, np.array(w_list), label=label_time[i])

    ax.set_ylabel('final_width (um)', fontsize=13)
    ax.set_xlabel('initial depth (um)', fontsize=13)
    ax.set_title('Widening against initial depth at different duration '
                 '\n etching rate = 2.3 um/min, width0 = 0.5um', fontsize=13)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(True)

    return w, ax


def width_t(w0, depth, rate, time):
    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1, 1)
    color = ['b', 'm', 'r', 'y']
    ax2 = plt.twinx(ax)
    for i, c in zip(range(len(w0)), color):
        w0_temp = w0[i]
        w_list = []
        dw_list = []
        for j in range(time):
            delta_w = width_cal(w0_temp / depth, rate[1], 1)
            w = delta_w + w0_temp
            w0_temp = w
            w_list.append(w)
        ax.plot(np.array([k for k in range(time)]), np.array(w_list), label=label_w0[i], color=c)
        dw_list = [w_list[i+1] - w_list[i] for i in range(len(w_list)-1)]
        ax2.plot(np.array([k for k in range(time-1)]), np.array(dw_list)*60, color=c, ls=':')

    ax.set_ylabel('width (um)', fontsize=13)
    ax.set_ylim(0, 30)
    ax.set_xlabel('time (s)', fontsize=13)
    ax2.set_ylabel('widening rate(um/min)', fontsize=13)
    ax2.set_ylim(0,5)
    ax.set_title('Widening against time '
                 '\n etching rate = 2.3 um/min, depth0=5um', fontsize=13)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    ax.legend(fontsize=13, loc=0)
    ax.grid(True)
    return ax


# w_2 = delta_width(w0, depth0, rate, time)

def quick_cal(w0, depth0, time, rate):

    w0_temp = w0
    for i in range(time):
        delta_w = width_cal(w0_temp / depth0, rate, 1)
        w = delta_w + w0_temp
        w0_temp = w

    return w