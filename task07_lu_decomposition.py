"""
BIT 2118 Scientific Computing - Task 7
Gaussian Elimination and LU Decomposition (scipy.linalg.lu_factor / lu_solve)
"""
import time
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------------
# A. MAIN CASE: small example, then many right-hand sides
# ------------------------------------------------------------------
print("=== A. MAIN CASE ===")
A = np.array([[2.0, 1.0, 1.0],
              [4.0, 3.0, 3.0],
              [8.0, 7.0, 9.0]])
P, L, U = lu(A)                                  # A = P L U
print("L =\n", L)
print("U =\n", U)
print("max |P L U - A| = %.2e" % np.abs(P @ L @ U - A).max())
b = np.array([4.0, 10.0, 24.0])
lu_piv = lu_factor(A)
print("solution of A x = b :", lu_solve(lu_piv, b))
print("np.linalg.solve     :", np.linalg.solve(A, b))


def compare(n, k, seed=0):
    """Solve A x = b_j for k different b_j: solve() each time vs one LU."""
    rng = np.random.default_rng(seed)
    M = rng.random((n, n)) + n * np.eye(n)
    B = rng.random((n, k))
    t0 = time.perf_counter()
    X1 = np.column_stack([np.linalg.solve(M, B[:, j]) for j in range(k)])
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    fac = lu_factor(M)                          # expensive step done once
    X2 = np.column_stack([lu_solve(fac, B[:, j]) for j in range(k)])
    t_lu = time.perf_counter() - t0
    return t_solve, t_lu, np.abs(X1 - X2).max()


print("\nn = 300, increasing number of right-hand sides k:")
ks = [1, 5, 20, 50, 100, 200]
ts, tl = [], []
for k in ks:
    a, b_, err = compare(300, k)
    ts.append(a)
    tl.append(b_)
    print("k = %3d | repeated solve %.4f s | LU once %.4f s | max diff %.1e" %
          (k, a, b_, err))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: matrix size n (k = 100 right-hand sides)
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: matrix size n ===")
for n in [100, 200, 400]:
    a, b_, _ = compare(n, 100)
    print("n = %3d | repeated solve %.4f s | LU once %.4f s | ratio x%.1f" %
          (n, a, b_, a / b_))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(ks, ts, "o-", label="np.linalg.solve each time")
ax.plot(ks, tl, "s-", label="lu_factor once + lu_solve")
ax.set_xlabel("Number of right-hand sides k (n = 300)")
ax.set_ylabel("Total time (s)")
ax.set_title("Task 7: reusing an LU factorisation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task07_lu_timing.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: steady heat conduction in a rod, 3 heat loads
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: heat in a 1 m rod with three heat-source cases ===")
n = 99                                      # interior nodes, ends held at 0 C
Lrod, kcond = 1.0, 50.0                     # length (m), conductivity (W/mK)
h = Lrod / (n + 1)
xs = np.linspace(h, Lrod - h, n)
K = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h ** 2   # -T'' matrix
fac = lu_factor(K)                          # factorise once
loads = {
    "uniform 1e4 W/m3": np.full(n, 1e4),
    "uniform 2e4 W/m3": np.full(n, 2e4),
    "localised centre source": 4e4 * np.exp(-((xs - 0.5) / 0.1) ** 2),
}
fig, ax = plt.subplots(figsize=(7, 3.8))
for name, q in loads.items():
    T = lu_solve(fac, q / kcond)            # T'' = -q/k
    print("%-26s max temperature rise = %.2f C" % (name, T.max()))
    ax.plot(xs, T, label=name)
exact = 1e4 * xs * (Lrod - xs) / (2 * kcond)
print("check vs analytic (uniform 1e4): max error = %.2e C" %
      np.abs(lu_solve(fac, np.full(n, 1e4) / kcond) - exact).max())
ax.set_xlabel("Position along rod (m)")
ax.set_ylabel("Temperature rise (C)")
ax.set_title("Task 7 (new case): rod temperature, one LU factorisation")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task07_rod.png", dpi=150)
plt.close(fig)
