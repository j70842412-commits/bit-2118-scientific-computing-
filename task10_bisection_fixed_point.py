"""
BIT 2118 Scientific Computing - Task 10
Root Finding: Bisection and Fixed-Point Methods (break-even and threshold values)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def bisection(f, a, b, tol=1e-6, maxit=100):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    hist = []
    for k in range(1, maxit + 1):
        c = (a + b) / 2
        hist.append(c)
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
        if (b - a) / 2 < tol:
            break
    return (a + b) / 2, k, hist


def fixed_point(g, x0, tol=1e-6, maxit=200):
    x = x0
    hist = [x]
    for k in range(1, maxit + 1):
        x_new = g(x)
        hist.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, k, hist
        x = x_new
    return x, maxit, hist


# ------------------------------------------------------------------
# A. MAIN CASE: break-even point  profit(x) = 0
#    price 50, variable cost 20 per unit, fixed cost 2000, extra cost 0.01 x^2
# ------------------------------------------------------------------
def make_profit(price):
    return lambda x: (price - 20.0) * x - 2000.0 - 0.01 * x ** 2


print("=== A. MAIN CASE: break-even units, price = KES 50 ===")
profit = make_profit(50.0)
x_b, it_b, hist_b = bisection(profit, 0, 200)
g = lambda x: (2000.0 + 0.01 * x ** 2) / 30.0      # rearranged: x = g(x)
x_f, it_f, hist_f = fixed_point(g, 100.0)
exact = (30 - np.sqrt(30 ** 2 - 4 * 0.01 * 2000)) / (2 * 0.01)
print("bisection   : x = %.6f in %d iterations" % (x_b, it_b))
print("fixed point : x = %.6f in %d iterations" % (x_f, it_f))
print("exact (quadratic formula): %.6f" % exact)
print("profit at root = %.6f" % profit(x_b))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: selling price 50 -> 45
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: price 50 -> 45 ===")
profit45 = make_profit(45.0)
g45 = lambda x: (2000.0 + 0.01 * x ** 2) / 25.0
xb45, ib45, _ = bisection(profit45, 0, 200)
xf45, if45, _ = fixed_point(g45, 100.0)
print("bisection   : x = %.6f in %d iterations" % (xb45, ib45))
print("fixed point : x = %.6f in %d iterations" % (xf45, if45))
print("break-even rises by %.2f units" % (xb45 - x_b))

xs = np.linspace(0, 150, 300)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(xs, profit(xs), label="price 50")
axs[0].plot(xs, profit45(xs), label="price 45")
axs[0].axhline(0, color="k", lw=0.8)
axs[0].plot([x_b, xb45], [0, 0], "ro")
axs[0].set_xlabel("Units sold x")
axs[0].set_ylabel("Profit (KES)")
axs[0].set_title("Break-even points")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].semilogy(np.abs(np.array(hist_b) - exact), label="bisection")
axs[1].semilogy(np.abs(np.array(hist_f[1:]) - exact) + 1e-16, label="fixed point")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("|error|")
axs[1].set_title("Convergence (price 50)")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task10_breakeven.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: when has a hot drink cooled to 45 C?
#    T(t) = T_env + (T0 - T_env) exp(-k t),  T0 = 90, T_env = 25, k = 0.08 /min
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: time for tea to cool to 45 C ===")
T0, Tenv, k, target = 90.0, 25.0, 0.08, 45.0
T = lambda t: Tenv + (T0 - Tenv) * np.exp(-k * t)
f = lambda t: T(t) - target
t_b, itb, hb = bisection(f, 0, 60)
relax = lambda t: t + 0.5 * (T(t) - target)         # fixed-point form with relaxation
t_f, itf, hf = fixed_point(relax, 5.0)
t_exact = np.log((T0 - Tenv) / (target - Tenv)) / k
print("bisection   : t = %.5f min in %d iterations" % (t_b, itb))
print("fixed point : t = %.5f min in %d iterations" % (t_f, itf))
print("exact       : t = %.5f min" % t_exact)

ts = np.linspace(0, 60, 300)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(ts, T(ts))
ax.axhline(target, color="r", ls="--", label="45 C")
ax.axvline(t_b, color="gray", ls=":", label="t = %.2f min" % t_b)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Temperature (C)")
ax.set_title("Task 10 (new case): cooling of a hot drink")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task10_cooling.png", dpi=150)
plt.close(fig)
