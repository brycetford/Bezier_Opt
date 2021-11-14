import numpy as np
import math
import time
import fun
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits import mplot3d

def loading_function():
    Cov = np.mat('2 0; 0 5')
    Loc = np.mat('0; 1')
    Rot = 0
    return fun.Gaussian2d(Cov, Loc, Rot)


def plot_curve(bezier, n):
    s_t = []
    for i in range(n):
        t = i / n
        s_t.append(bezier.eval(t))
    s_xy = np.array([s_t]).T
    plt.scatter(s_xy[:, 0], s_xy[:, 1])
    plt.show()


def eval_load(path, load_fun, n):
    load = 0
    li = load_fun.eval(path.eval(0))

    for i in range(n - 1):
        dt = (i + 1) / n
        lf = load_fun.eval(path.eval(dt)) * np.linalg.norm(path.d_dt(dt))
        load += dt * (li + lf) / 2  # Center
        li = lf

    return load


def delta_curve(curvature):
    if curvature > 0.5:
        return 1
    return 0


def eval_curve(path, n):
    lam = 25
    curv = 0
    for i in range(n):
        dt = i/n
        curv += path.curvature(dt) * np.linalg.norm(path.d_dt(dt)) * delta_curve(path.curvature(dt))
    return math.exp(curv/lam)


def gradient(load_fun, curr_load, curr_curv, params, n):
    h = 0.1  # How far to 'nudge' each parameter in gradient computation
    size = params.shape

    d_dx = params + np.array([0, h, 0, 0, 0, 0]).reshape(size)
    d_dy = params + np.array([0, 0, 0, 0, h, 0]).reshape(size)

    d_dBx = fun.Bezier(2, 3, d_dx)
    d_dBy = fun.Bezier(2, 3, d_dy)

    dL_dBx = (eval_load(d_dBx, load_fun, n) - curr_load) / h
    dL_dBy = (eval_load(d_dBy, load_fun, n) - curr_load) / h

    dC_dBx = (eval_curve(d_dBx, n) - curr_curv) / h
    dC_dBy = (eval_curve(d_dBy, n) - curr_curv) / h

    return np.array([0, dL_dBx + dC_dBx, 0, 0, dL_dBy + dC_dBy, 0]).reshape(size)


def visualize_cost(load_fun, n):
    r_min, r_max = -1, 3
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    x, y = np.meshgrid(xaxis, yaxis)
    results = load_wrapper(x, y, load_fun, n) + curve_wrapper(x, y, n)
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    pyplot.show()


def load_wrapper(x, y, load_fun, n):
    results = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Params = np.mat([0, x[i, j], 2, 0, y[i, j], 2]).reshape(2, 3)
            B = fun.Bezier(2, 3, Params)
            results[i, j] = eval_load(B, load_fun, n)

    return results


def curve_wrapper(x, y, n):
    results = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Params = np.mat([0, x[i, j], 2, 0, y[i, j], 2]).reshape(2, 3)
            B = fun.Bezier(2, 3, Params)
            results[i, j] = eval_curve(B, n)

    return results


if __name__ == "__main__":
    # Initial guess for path
    Params = np.mat('[0 0; 1 1; 2 2]').T  # Initial control points fot the Bezier curve
    B = fun.Bezier(2, 3, Params)

    n = 40  # Iterations for numerical computations

    # Loading function
    Load = loading_function()
    visualize_cost(Load, n)

    alpha = 0.05  # Gradient descent rate

    plot_curve(B, n)

    for i in range(n):
        Curr_Load = eval_load(B, Load, n)
        Curr_Curv = eval_curve(B, n)
        Params = Params - alpha * gradient(Load, Curr_Load, Curr_Curv, Params, n)
        B = fun.Bezier(2, 3, Params)

    plot_curve(B, n)

    print(Params)
