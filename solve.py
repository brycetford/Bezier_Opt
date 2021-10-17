import numpy as np
import math
import fun

def loading_function():
    Cov = np.mat('2 0; 0 5')
    Loc = np.mat('2; 2')
    Rot = math.pi / 4
    return fun.Gaussian2d(Cov, Loc, Rot)

def eval_load(path, load_fun, n):

    s_t = []
    for i in range(n):
        t = i/n
        s_t.append(path.eval(t))

    load = 0
    for i in range(n-1):
        u = load_fun.eval(s_t[i])
        h = load_fun.eval(s_t[i+1])
        d_l = (u+h)/2
        load += d_l

    return load


if __name__ == "__main__":
    # Initial guess for path
    Params = np.mat('[0 0; 2 2; 4 4]').T
    B = fun.Bezier(2, 3, Params)

    Load = loading_function()

    curr_load = eval_load(B, Load, 100)
    print(curr_load)