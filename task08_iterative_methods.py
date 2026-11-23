"""
BIT 2118 Scientific Computing - Task 8
Iterative Methods for Linear Systems (Jacobi and Gauss-Seidel)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def jacobi(A, b, rtol=1e-8, maxit=50000):
    D = np.diag(A)
    R = A - np.diagflat(D)
    x = np.zeros_like(b)
    hist = []
    for k in range(1, maxit + 1):
        x = (b - R @ x) / D
        r = np.linalg.norm(b - A @ x)
        hist.append(r)
        if r < rtol * np.linalg.norm(b):
            break
    return x, k, hist


def gauss_seidel(A, b, rtol=1e-8, maxit=50000):
    n = len(b)
    x = np.zeros_like(b)
    hist = []
    for k in range(1, maxit + 1):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - s) / A[i, i]          # uses newest values immediately
        r = np.linalg.norm(b - A @ x)
        hist.append(r)
        if r < rtol * np.linalg.norm(b):
            break
    return x, k, hist


def rod_system(n, T_left=100.0, T_right=0.0):
    """Steady 1-D heat conduction: n interior nodes, fixed end temperatures."""
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.zeros(n)
    b[0], b[-1] = T_left, T_right
    return A, b


# ------------------------------------------------------------------
# A. MAIN CASE: 10-node rod, ends at 100 C and 0 C
# ------------------------------------------------------------------
print("=== A. MAIN CASE: heat conduction in a rod (n = 10) ===")
A, b = rod_system(10)
exact = np.linalg.solve(A, b)
xj, kj, hj = jacobi(A, b)
xg, kg, hg = gauss_seidel(A, b)
print("exact (direct solve):", exact)
print("Jacobi       : %d iterations, max error %.2e" % (kj, np.abs(xj - exact).max()))
print("Gauss-Seidel : %d iterations, max error %.2e" % (kg, np.abs(xg - exact).max()))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: number of nodes n (10 -> 20 -> 40)
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: system size n ===")
for n in [10, 20, 40]:
    A_, b_ = rod_system(n)
    _, kj_, _ = jacobi(A_, b_)
    _, kg_, _ = gauss_seidel(A_, b_)
    print("n = %2d | Jacobi %5d iterations | Gauss-Seidel %5d iterations" %
          (n, kj_, kg_))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.semilogy(hj, label="Jacobi")
ax.semilogy(hg, label="Gauss-Seidel")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual norm ||b - A x||")
ax.set_title("Task 8: convergence of iterative methods (n = 10)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task08_convergence.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: node voltages of a 4-node resistor network
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: node voltages in an electrical network ===")
G = np.array([[4.0, -1.0, -1.0, 0.0],        # conductance matrix (siemens)
              [-1.0, 4.0, 0.0, -1.0],
              [-1.0, 0.0, 4.0, -1.0],
              [0.0, -1.0, -1.0, 4.0]])
I_src = np.array([10.0, 0.0, 0.0, 5.0])      # injected currents (amperes)
print("strictly diagonally dominant:", bool(np.all(2 * np.abs(np.diag(G)) >
                                                   np.abs(G).sum(axis=1))))
v_exact = np.linalg.solve(G, I_src)
vg, kg2, hg2 = gauss_seidel(G, I_src, rtol=1e-10)
vj, kj2, hj2 = jacobi(G, I_src, rtol=1e-10)
print("direct solution V (V):", v_exact)
print("Gauss-Seidel V (V)   :", vg, "in %d iterations" % kg2)
print("Jacobi V (V)         :", vj, "in %d iterations" % kj2)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.semilogy(hj2, label="Jacobi")
ax.semilogy(hg2, label="Gauss-Seidel")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual norm")
ax.set_title("Task 8 (new case): network node voltages")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task08_network.png", dpi=150)
plt.close(fig)
