import numpy as np
import matplotlib.pyplot as plt

class System:

    def __init__(self, x, x_dot, mean, cov):
        self.init = np.mat([x_dot, x]).T
        self.state = self.init
        self.state_mat = np.mat('1 0; 0 -1')

        self.mu = mean
        self.sigma = cov

    def input(self):
        f = 5*np.random.normal(self.mu, self.sigma, 1)
        return f

    def iterate(self):
        dt = 0.01
        f = np.mat([0, self.input()]).T
        x_dot = self.state_mat.dot(self.state) + f
        self.state = dt*np.mat('0 1; 1 0').dot(x_dot) + self.state


def main():
    n = 5000
    x_t = [0]*n
    t = [0]*n
    x = System(0, 0, 0, 1)
    for i in range(n):
        t[i] = i
        x.iterate()
        x_t[i] = x.state[1]
    plt.scatter(t, x_t)
    plt.show()


if __name__ == "__main__":
    main()
