import numpy as np
import math
import fun
import matplotlib.pyplot as plt

def loading_function():
    Cov = np.mat('2 0; 0 5')
    Loc = np.mat('1; 1')
    Rot = 0
    return fun.Gaussian2d(Cov, Loc, Rot)

def eval_load(path, load_fun, n):

    load = 0
    li = load_fun.eval(path.eval(0))

    for i in range(n-1):
        dt = (i+1)/n
        lf = load_fun.eval(path.eval(dt))
        load += dt*(li + lf)/2     # Center
        li = lf

    return load

def gradient(load_fun, curr_load, params, n):

    h = 0.1     # How far to 'nudge' each parameter in gradient computation
    size = params.shape

    d_dx = params + np.array([0, h, 0, 0, 0, 0]).reshape(size)
    d_dy = params + np.array([0, 0, 0, 0, h, 0]).reshape(size)

    d_dBx = fun.Bezier(2, 3, d_dx)
    d_dBy = fun.Bezier(2, 3, d_dy)

    dL_dBx = (eval_load(d_dBx, load_fun, n) - curr_load) / h
    dL_dBy = (eval_load(d_dBy, load_fun, n) - curr_load) / h

    return np.array([0, dL_dBx, 0, 0, dL_dBy, 0]).reshape(size)


if __name__ == "__main__":
    # Initial guess for path
    Params = np.mat('[0 0; 1 1; 2 2]').T    # Initial control points fot the Bezier curve
    B = fun.Bezier(2, 3, Params)

    # Loading function
    Load = loading_function()

    alpha = 0.05    # Gradient descent rate
    n = 20  # Iterations for numerical computations

    for i in range(5):
        Curr_Load = eval_load(B, Load, n)
        Params = Params -alpha*gradient(Load, Curr_Load, Params, n)
        B = fun.Bezier(2, 3, Params)

    print(Params)