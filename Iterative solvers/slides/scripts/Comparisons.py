import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, tril, triu, eye
from scipy.sparse.linalg import spsolve_triangular, factorized

from tqdm import trange

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
    residu = [1]
    for _ in trange(maxiter):
        if (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) < tol:
            break
        x -= invD * Δx
        residu.append(res)

    return np.asarray(residu)

def gauss_seidel(A, b, tol=1e-10, maxiter=500000):

    # --> Initial condition.
    x = np.zeros_like(b)

    # --> Partition the matrix.
    L = tril(A).tocsr()
    solver = factorized(L)

    # --> Gauss-Seidel iteration.
    residu = [1]
    for _ in trange(maxiter):
        if (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) < tol:
            break
        x -= solver(Δx)
        residu.append(res)

    return np.asarray(residu)

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
    for k in trange(maxiter):
        res = np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b)
        residu.append(res)

        if (res < tol) and (k % m == 0):
            break
        x -= ω[k % m]* invD * Δx

    return np.asarray(residu)

if __name__ == "__main__":

    n = 256
    L = laplacian_1D(n)
    b = np.ones(n)

    jacobi_residual = jacobi_solver(L, b)
    gauss_seidel_residual = gauss_seidel(L, b)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    ax.semilogy(np.arange(0, len(jacobi_residual))/1000, jacobi_residual, color="black", label="Jacobi")

    ax.set(ylim=(1e-10, 10), ylabel="Residual")
    ax.set(xlim=(0, 400), xlabel="Iterations x 1000")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right")
    plt.savefig("../imgs/comparaisons_1D_jacobi.png", bbox_inches="tight", dpi=1200)

    ax.semilogy(np.arange(0, len(gauss_seidel_residual))/1000, gauss_seidel_residual, color="red", label="Gauss-Seidel")
    ax.legend(loc="upper right")
    plt.savefig("../imgs/comparaisons_1D_gs.png", bbox_inches="tight", dpi=1200)

    K = [2, 4, 8, 16, 32, 64, 128, 256]
    for k in K:
        ω = get_omega(k, n)

        srj_residual = srj_solver(L, b, ω)

        if k == 2:
            ax.semilogy(np.arange(0, len(srj_residual))[::k]/1000, srj_residual[::k], color="dodgerblue", label="SRJ", ls='--')
            ax.legend(loc="upper right")
        else:
            ax.semilogy(np.arange(0, len(srj_residual))[::k]/1000, srj_residual[::k], color="dodgerblue", label="SRJ")

        plt.savefig("../imgs/comparaisons_1D_srj_k={k}.png".format(k=k), bbox_inches="tight", dpi=1200)

    ax.set_xlim(1, 400)
    ax.set_xscale("log")
    plt.savefig("../imgs/comparaisons_1D_logscale.png".format(k=k), bbox_inches="tight", dpi=1200)






    n = 512
    L = laplacian_2D(n)

    x = np.linspace(0, 1, n+2)[1:-1] - 0.5
    x, y = np.meshgrid(x, x)
    θ = np.pi/4
    x, y = np.cos(θ)*x + np.sin(θ)*y, -np.sin(θ)*x + np.cos(θ)*y
    b = np.exp(-x**2/0.005 - y**2/0.1)
    x, y = np.cos(θ)*x - np.sin(θ)*y, np.sin(θ)*x + np.cos(θ)*y
    b = b.flatten()

    jacobi_residual = jacobi_solver(L, b)
    gauss_seidel_residual = gauss_seidel(L, b)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    
    ax.semilogy(np.arange(0, len(jacobi_residual))/1000, jacobi_residual, color="black", label="Jacobi")
    
    ax.set(ylim=(1e-10, 10), ylabel="Residual")
    ax.set(xlim=(0, 500), xlabel="Iterations x 1000")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()
    plt.savefig("../imgs/comparaisons_2D_jacobi.png", bbox_inches="tight", dpi=1200)
    
    ax.semilogy(np.arange(0, len(gauss_seidel_residual))/1000, gauss_seidel_residual, color="red", label="Gauss-Seidel")
    ax.legend()
    plt.savefig("../imgs/comparaisons_2D_gs.png", bbox_inches="tight", dpi=1200)

    K = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for k in K:
        ω = get_omega(k, n)

        srj_residual = srj_solver(L, b, ω)

        if k == 2:
            ax.semilogy(np.arange(0, len(srj_residual))[::k]/1000, srj_residual[::k], color="dodgerblue", label="SRJ", ls='--')
            ax.legend()
        else:
            ax.semilogy(np.arange(0, len(srj_residual))[::k]/1000, srj_residual[::k], color="dodgerblue", label="SRJ")

        plt.savefig("../imgs/comparaisons_2D_srj_k={k}.png".format(k=k), bbox_inches="tight", dpi=1200)

    ax.set_xlim(1, 500)
    ax.set_xscale("log")
    plt.savefig("../imgs/comparaisons_2D_logscale.png".format(k=k), bbox_inches="tight", dpi=1200)
    plt.show()
