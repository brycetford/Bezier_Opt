import numpy as np
import math

class Bezier:

    # Create a bezier curve with a specified order(+1) and dimension, and initialize its parameters
    def __init__(self, dim, order, params):

        # Raise an exception if invalid dimensions
        if (dim != params.shape[0]) | (order != params.shape[1]):
            raise SyntaxError('Parameter dimensions do not match given dimensions')

        self.dim = dim  # Dimension i.e. x,y,z...
        self.order = order  # Order, +1 of the order of the actual curve
        self.params = params    # Control points of the curve as a numpy matrix

    # Evaluate the bezier curve at given t (0 <= t <= 1)
    def eval(self, t):

        result = np.zeros((self.dim, 1))

        for i in range(self.order):
            c = math.comb(self.order, i) * ((1-t)**(self.order-i)) * t**i
            result = result + c*self.params[:, i]

        return result

class Gaussian2d:

    # Create a Gaussian distribution in 2d with a given covariance matrix
    def __init__(self, cov, loc, rot):

        # Raise an exception if the covariance matrix has invalid dimensions
        if cov.shape != (2, 2):
            raise SyntaxError('Requires 2x2 covariance matrix')
        if loc.shape != (2, 1):
            raise SyntaxError('Requires 1x2 location vector (as a np matrix 1x2)')

        # 2d rotation matrix
        R = np.array([math.cos(rot), math.sin(rot), -math.sin(rot), math.cos(rot)]).reshape((2, 2))

        self.cov = (R.dot(cov)).dot(R.T)    # Rotate cov by rot
        self.loc = loc

    def eval(self, x):
        dist = x-self.loc
        arg = (dist.T.dot(self.cov)).dot(dist)
        return math.exp(-0.5 * arg)


# Testing
if __name__ == "__main__":
    # Bezier
    Params = np.mat('[1 2; 3 4; 6 7]').T
    B = Bezier(2, 3, Params)
    print(B.eval(0.4))
    print("Bezier Working")

    # Gaussian2d
    Cov = np.mat('2 0; 0 5')
    Loc = np.mat('1; 1')
    Rot = math.pi/4
    gauss = Gaussian2d(Cov, Loc, Rot)
    X = np.mat('0.5; 0.5')
    print(gauss.eval(X))
    print("Gaussian2d Working")

