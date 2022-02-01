import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve_triangular, factorized

from scipy.integrate import solve_ivp

def laplacian_1D(n):

    dx = 1 / (n+1)

    d = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]

    L = diags(d, (-1, 0, 1)) / dx**2

    return L.tocsr()

def laplacian_2D(n):

    Lx = laplacian_1D(n)
    Ly = laplacian_1D(n)

    I = eye(n)

    L = kron(Lx, I) + kron(I, Ly)

    return L.tocsr()

def main():

    # --> Define the matrix in the problem.
    n = 256
    L = laplacian_2D(n)

    # --> Define the rhs of the heat equation.
    def rhs(t, u):
        return L @ u

    # --> Parameters of the simulation.
    tspan = (0.0, 10.0)
    t = np.linspace(*tspan, 1024)

    return L

if __name__ == "__main__":
    L = main()
