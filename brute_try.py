import numpy as np
from scipy.optimize import least_squares, brute
import scipy
from scipy import stats
import matplotlib.pyplot as plt


params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)


def f1(z, *params):
    x, y, w, v = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)


def f2(z, *params):
    x, y, w, v = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-g*np.exp(-((x-h)**2 + (y-i)**2 + (w-j)**2) + (v-g)**2 / scale))


def f3(z, *params):
    x, y, w, v = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-j*np.exp(-((x-k)**2 + (y-l)**2 + (w-j)**2) + (v-g)**2 / scale))


def f(z, *params):
    return f1(z, *params) + f2(z, *params) + f3(z, *params)


rranges = (slice(-4, 4, 0.1), slice(-4, 4, 0.1), slice(-4, 4, 0.2), slice(-2, 2, 1))
resbrute = brute(f, rranges, args=params, full_output=True, finish=None)

# z=(1,2)
# h= f3(z, *params)
# k=f(z, *params)
