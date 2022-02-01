import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, tril, triu, eye
from scipy.sparse.linalg import spsolve_triangular, factorized

from tqdm import trange, tqdm

def laplacian_1D(n):
    # --> Step size.
    Δx = 1 / (n+1)

    # --> Diagonals.
    d = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]

    # --> Construct Laplacian matrix.
    L = diags(d, (-1, 0, 1)) #/ Δx**2

    return L.tocsr()

def laplacian_2D(n):

    Lx = laplacian_1D(n)
    Ly = laplacian_1D(n)

    I = eye(n)

    L = kron(Lx, I) + kron(I, Ly)

    return L.tocsr()

def jacobi_solver(A, b, tol=1e-10, maxiter=500000):
    """Simple implementation of the Jacobi method."""
    # --> Initial condition.
    x = np.zeros_like(b)

    # --> Partition the matrix.
    # D = A.diagonal()
    # invD = diags(1.0/D)
    invD = 1 / A[0, 0]

    # --> Jacobi iteration.
    i = 0
    while (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) > tol:
        x -= invD * Δx
        i += 1

    return i

def gauss_seidel(A, b, tol=1e-10, maxiter=500000):

    # --> Initial condition.
    x = np.zeros_like(b)

    # --> Partition the matrix.
    L = tril(A).tocsr()
    solver = factorized(L)

    # --> Gauss-Seidel iteration.
    residu = [1]
    i = 0
    while (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) > tol:
        x -= solver(Δx)
        i += 1

    return i

def order_omega(ω, kmin, kmax, n):
    ωo = np.zeros_like(ω)
    q = np.ones_like(ωo)
    M = np.sum(q)
    k = np.arange(kmin, kmax, kmin)
    G = np.ones_like(k)

    ωo[0] = ω[0]
    q[0] = q[0] -1
    G = G * np.abs(1 - k*ωo[1])

    counter = 1

    while sum(q != 0):
        if sum(q == 0) != 0:
            idx = np.argwhere(q == 0)[0, 0]
            if idx == 0:
                ω = ω[1:]
                q = q[1:]
            elif idx == len(q)-1:
                ω = ω[:-1]
                q = q[:-1]
            else:
                ω[idx:-1] = ω[idx+1:]
                ω = ω[:-1]
                q[idx:-1] = q[idx+1:]
                q = q[:-1]

        ww = 1 / k[np.argmax(G)]
        dis = np.abs(ω - ww)
        idx = np.argmin(dis)
        ωo[counter] = ω[idx]
        G = G * np.abs(1 - k * ωo[counter])
        q[idx] = q[idx] - 1
        counter += 1

    return ωo

def get_omega(m, n):

    kmax, kmin = 2, 2*np.sin(np.pi/(2*n))**2

    k = np.arange(1, m+1)
    ω = np.cos(np.pi * (2*k-1)/(2*m))

    ω = kmax + kmin - (kmax-kmin)*ω
    ω = 2/ω

    return order_omega(ω, kmin, kmax, n)

def srj_solver(A, b, ω, maxiter=500000, tol=1e-10):
    """Simple implementation of the Jacobi method."""
    # --> Initial condition.
    x = np.zeros_like(b)
    m = len(ω)

    # --> Partition the matrix.
    # D = A.diagonal()
    # invD = diags(1.0/D)
    invD = 1 / A[0, 0]

    # --> Jacobi iteration.
    residu = [1]
    k = 0
    while (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) > tol:
        x -= ω[k % m]* invD * Δx
        k += 1

    return k

if __name__ == "__main__":

    N = np.array([32, 64, 128, 256, 512, 1024, 2048])
    K = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    iter_jacobi = np.zeros(len(N))
    iter_gs = np.zeros(len(N))
    iter_srj = np.zeros((len(N), len(K)))

    for i, n in enumerate(tqdm(N)):
        L = laplacian_1D(n)
        b = np.ones(n)

        print("Running Jacobi solver...")
        iter_jacobi[i] = jacobi_solver(L, b)

        print("Running Gauss-Seidel solver...")
        iter_gs[i] = gauss_seidel(L, b)

        for j, k in enumerate(K):
            ω = get_omega(k, n)
            print("Running SJR solver with k=", k, "...")
            iter_srj[i, j] = srj_solver(L, b, ω)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot(N, iter_jacobi.reshape(-1, 1)/iter_srj)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set(xticks=N, xticklabels=N)
    ax.set(xlim=(N.min(), N.max()), ylim=(1, 1024))
    ax.set(xlabel="Number of grid points", ylabel="Speed-up")
    ax.grid(True, which="major", axis="y")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig("Performances_1D.png", bbox_inches="tight", dpi=300)
    plt.show()
