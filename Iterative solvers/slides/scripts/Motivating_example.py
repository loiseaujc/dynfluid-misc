import numpy as np

import matplotlib.pyplot as plt

def jacobi_solver(A, b, tol=1e-10):
    """Simple implementation of the Jacobi method."""
    # --> Initial condition.
    x = np.zeros_like(b)

    # --> Partition the matrix.
    D = np.diag(np.diag(A))
    R = A - D

    # --> Jacobi iteration.
    residu = list()
    while (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) > tol:
        x -= np.linalg.solve(D, Δx)
        residu.append(res)

    return np.asarray(residu)

def gauss_seidel(A, b, tol=1e-10):

    # --> Initial condition.
    x = np.zeros_like(b)

    # --> Partition the matrix.
    L = np.tril(A)
    R = A - L

    # --> Gauss-Seidel iteration.
    residu = list()
    while (res := np.linalg.norm(Δx := A @ x - b) / np.linalg.norm(b) ) > tol:
        x -= np.linalg.solve(L, Δx)
        residu.append(res)

    return np.asarray(residu)


if __name__ == "__main__":

    # --> Define the problem.
    A = np.array([[-2.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -2.0]])
    b = np.random.rand(3)

    # --> Jacobi solver.
    jacobi_residual = jacobi_solver(A, b)

    # --> Gauss-Seidel solver.
    gauss_seidel_residual = gauss_seidel(A, b)

    # -->
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))

    ax.semilogy(jacobi_residual, color="black", label="Jacobi")
    ax.semilogy(gauss_seidel_residual, color="red", label="Gauss-Seidel")

    ax.set(ylim=(1e-10, 100), ylabel="Residual")
    ax.set(xlim=(0, 80), xlabel="Iteration")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axvline(3, c="lightgray", ls="--", lw=1)

    ax.legend()

    plt.savefig("../imgs/motivating_example.png", bbox_inches="tight", dpi=1200)
    plt.show()
