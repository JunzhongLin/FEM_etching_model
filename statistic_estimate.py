import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
from wave_model import width_cal
from scipy.optimize import least_squares, brute


path = r'./data'
sheet = 'worksheet.xlsx'


def read_data():
    return pd.read_excel(os.path.join(path, sheet))


data = read_data()

rate = 2.3/60
time = np.array([1, 2, 3, 4, 5, 6, 7, 10]) * 60
size = 10000
bin_size = 100


# R = stats.truncnorm.rvs((0-loc)/scale, (100-loc)/scale, size=size, loc=loc, scale=scale)

def width_plot(data):
    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            if i == 0:
                r = ax[i][j].hist(data.iloc[:, j].dropna().values, density=True)
                param = stats.norm.fit(data.iloc[:, j].dropna().values)
                fitted_data = stats.norm.pdf(r[1], loc=param[0], scale=param[1])
                ax[i][j].plot(r[1], fitted_data)
                ax[i][j].set_xlim(0, 20)
                ax[i][j].set_title(data.columns.values[j])
            else:
                r = ax[i][j].hist(data.iloc[:, j+4].dropna().values, density=True)
                param = stats.norm.fit(data.iloc[:, j+4].dropna().values)
                fitted_data = stats.norm.pdf(r[1], loc=param[0], scale=param[1])
                ax[i][j].plot(r[1], fitted_data)
                ax[i][j].set_xlim(0, 20)
                ax[i][j].set_title(data.columns.values[j+4])

    return ax


def width_plot_polish(data):
    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(1, 4)
    for j in range(4):
        r = ax[j].hist(data.iloc[:, j].dropna().values, density=True)
        param = stats.lognorm.fit(data.iloc[:, j].dropna().values)
        fitted_data = stats.lognorm.pdf(r[1], s=param[0], loc=param[1], scale=param[2])
        ax[j].plot(r[1], fitted_data)
        ax[j].set_xlim(0, 25)
        ax[j].set_ylim(0, 0.5)
        ax[j].set_title(data.columns.values[j])

    return ax


def delta_width(w0, R, rate, time, size):
    w = 0
    plt.rcParams['savefig.dpi'] = 1200
    fig, ax = plt.subplots(2, 4)

    for i in range(len(time)):
        w_list = []
        w0_temp = np.ones(size) * w0
        depth0 = w0 / R
        for k in range(time[i]):
            delta_w = width_cal(w0_temp/depth0, rate, 1)
            w = delta_w + w0_temp
            w0_temp = w
        w_list.append(w)
        print(w_list)
        r = ax[i//4][i % 4].hist(w_list[0], density=True, bins=100)
        param = stats.norm.fit(w_list[0])
        fitted_data = stats.norm.pdf(r[1], loc=param[0], scale=param[1])
        ax[i // 4][i % 4].plot(r[1], fitted_data)
        ax[i // 4][i % 4].set_xlim(0, 20)
        ax[i // 4][i % 4].set_title(str(time[i]/60) + 'min')

    return ax


class WidthFitting:

    def __init__(self, data, rate, time, size, bin_size):  # bin_size to generate x0, y1, y0
        self.data = data
        self.rate = rate
        self.time = time
        self.size = size                              # calculate the width
        self.bin_size = bin_size   # bin_size to generate x0, y1, y0
        self.para0 = self.gen_para0()

    def gen_para0(self):
        para0 = np.zeros([8, 2])
        for i in range(8):
            para_temp = stats.norm.fit(self.data.iloc[:, i].dropna().values)
            para0[i, 0] = para_temp[0]
            para0[i, 1] = para_temp[1]
        return para0

    def gen_x0(self, w_list):
        x0 = np.zeros([len(self.time), self.bin_size])
        for i in range(len(self.time)):
            x0[i] = np.histogram(w_list[i], bins=self.bin_size, density=True)[1][:-1]
        return x0

    def gen_y1(self, w_list):
        y1 = np.zeros([len(self.time), self.bin_size])
        for i in range(len(self.time)):
            y1[i] = np.histogram(w_list[i], bins=self.bin_size, density=True)[0]
        return y1

    def cal_width(self, w0, loc, scale):

        R = stats.truncnorm.rvs((0-loc)/scale, (5-loc)/scale,         # Truncnorm distibuiion
                                size=self.size, loc=loc, scale=scale, random_state=1234)

        # R = stats.halflogistic.rvs(size=self.size, loc=loc, scale=scale, random_state=1234)
        w_list = np.zeros([len(self.time), self.size])

        for i in range(len(self.time)):
            w0_temp = np.ones(self.size) * w0
            depth0 = w0 / R
            for k in range(self.time[i]):
                delta_w = width_cal(w0_temp / depth0, rate, 1)
                w = delta_w + w0_temp
                w0_temp = w
            w_list[i] = w

        return w_list

    def cost_func(self, opt_x, f_list):   # opt_x = [w0, loc, scale]
        y0 = np.zeros([len(f_list), self.bin_size])
        opt_1, opt_2, opt_3 = opt_x
        w_list = self.cal_width(opt_1, opt_2, opt_3)
        x0 = self.gen_x0(w_list)
        y1 = self.gen_y1(w_list)

        for i in range(len(f_list)):
            y0[i] = stats.norm.pdf(x0[i], loc=self.para0[f_list[i], 0], scale=self.para0[f_list[i], 1])

        # return y1[f_list].flatten() - y0.flatten()
        return (sum((y1[f_list].flatten() - y0.flatten())**2)**0.5/y0.flatten().shape[0])

    def brute_search(self, f_list):
        rrange = (slice(0.1, 5, 1), slice(0.1, 2, 0.25), slice(0.1, 5, 1))
        resbrute = brute(self.cost_func, rrange, args=([f_list],), full_output=True)
        return resbrute

    def ls_fitting(self, opt_x0, f_list):
        res_1 = least_squares(self.cost_func, opt_x0, args=([f_list]),
                              bounds=([0, 0, 0], [5, 5, 15]), loss='soft_l1')
        return res_1

    def myplot(self, res_1, f_list):
        plt.rcParams['savefig.dpi'] = 1200
        fig, ax = plt.subplots(2, 4)
        w_list = self.cal_width(res_1.x[0], res_1.x[1], res_1.x[2])
        for i in range(len(time)):
            x_raw = np.histogram(self.data.iloc[:, i].dropna().values)[1]
            y_raw = stats.norm.pdf(x_raw, loc=self.para0[i, 0], scale=self.para0[i, 1])
            ax[i // 4][i % 4].plot(x_raw, y_raw, color='g')
            if i in f_list:
                ax[i // 4][i % 4].hist(w_list[i], color='r', bins=self.bin_size, density=True)
                ax[i // 4][i % 4].set_title(str(time[i] / 60) + 'min Fitted')
            else:
                ax[i // 4][i % 4].hist(w_list[i], color='b', bins=self.bin_size, density=True)
                ax[i // 4][i % 4].set_title(str(time[i] / 60) + 'min')

            ax[i // 4][i % 4].set_xlim(0, 20)
            ax[i // 4][i % 4].set_ylim(0, 0.55)

        return ax


# w0 = 2.0936138
# loc = 0.11499687
# scale = 1.74547927
# opt_x = [w0, loc, scale]
# wf = WidthFitting(data, rate, time, size, bin_size)
# f_list = [3]
# # res_brute = wf.brute_search(f_list)
#
# res = wf.ls_fitting(opt_x, f_list)
# wf.myplot(res, f_list)


