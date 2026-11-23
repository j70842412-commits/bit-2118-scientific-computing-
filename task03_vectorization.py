"""
BIT 2118 Scientific Computing - Task 3
Mathematical Functions and Vectorization (loop versus NumPy, timed with timeit)
"""
import math
import timeit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def f_loop(x_list):
    """f(x) = sin(x) + 0.5 x^2 computed one element at a time."""
    out = []
    for v in x_list:
        out.append(math.sin(v) + 0.5 * v * v)
    return out


def f_vec(x):
    """Same function applied to the whole array at once."""
    return np.sin(x) + 0.5 * x ** 2


def best_time(func, arg, n_calls):
    return timeit.timeit(lambda: func(arg), number=n_calls) / n_calls


# ------------------------------------------------------------------
# A. MAIN CASE: f(x) = sin(x) + 0.5 x^2, loop vs vectorised
# ------------------------------------------------------------------
print("=== A. MAIN CASE: f(x) = sin(x) + 0.5*x**2 ===")
sizes = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
t_loop, t_vec = [], []
for n in sizes:
    x = np.linspace(0, 10, n)
    xl = x.tolist()
    calls = max(1, 200000 // n)
    tl = best_time(f_loop, xl, calls)
    tv = best_time(f_vec, x, calls)
    assert np.allclose(f_loop(xl), f_vec(x))         # both give the same answer
    t_loop.append(tl)
    t_vec.append(tv)
    print("N = %8d | loop %.6f s | numpy %.6f s | speed-up x%.0f" %
          (n, tl, tv, tl / tv))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: number of elements N (1,000 -> 1,000,000)
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: effect of N on speed-up ===")
for n, tl, tv in zip(sizes, t_loop, t_vec):
    print("N = %8d -> speed-up x%.1f" % (n, tl / tv))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.loglog(sizes, t_loop, "o-", label="Python loop")
ax.loglog(sizes, t_vec, "s-", label="NumPy vectorised")
ax.set_xlabel("Number of elements N")
ax.set_ylabel("Time per run (s)")
ax.set_title("Task 3: loop vs vectorised execution time")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task03_timing.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: 1,000,000 mobile-money style transactions
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: KES -> USD and VAT on 1,000,000 transactions ===")
rng = np.random.default_rng(42)
amounts = rng.lognormal(mean=7.0, sigma=1.0, size=1_000_000)   # KES, VAT-inclusive
rate = 129.0                  # assumed KES per USD
vat_rate = 0.16               # Kenyan standard VAT rate


def convert_loop(a):
    usd, vat = [], []
    for v in a:
        usd.append(v / rate)
        vat.append(v * vat_rate / (1 + vat_rate))
    return usd, vat


def convert_vec(a):
    return a / rate, a * vat_rate / (1 + vat_rate)


a_list = amounts.tolist()
tl = best_time(convert_loop, a_list, 1)
tv = best_time(convert_vec, amounts, 5)
usd, vat = convert_vec(amounts)
print("total value  : KES %.2f million" % (amounts.sum() / 1e6))
print("total in USD : USD %.2f thousand" % (usd.sum() / 1e3))
print("total VAT    : KES %.2f million" % (vat.sum() / 1e6))
print("loop %.4f s | numpy %.4f s | speed-up x%.0f" % (tl, tv, tl / tv))
