"""
BIT 2118 Scientific Computing - Task 20
Random Numbers and Monte Carlo Methods (estimating pi, an integral, project risk)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# A. MAIN CASE: estimate pi and the integral of exp(-x^2) over [0, 1]
# ------------------------------------------------------------------
print("=== A. MAIN CASE (seed 42, N = 100000) ===")
rng = np.random.default_rng(42)
N = 100_000
px, py = rng.random(N), rng.random(N)
inside = px ** 2 + py ** 2 <= 1.0
pi_est = 4 * inside.mean()
print("pi estimate = %.5f | true %.5f | error %.5f" % (pi_est, np.pi, abs(pi_est - np.pi)))

xs = rng.random(N)
g = np.exp(-xs ** 2)
I_est = g.mean()
se = g.std(ddof=1) / np.sqrt(N)
I_true = 0.7468241328124270                      # known value (via error function)
print("integral of exp(-x^2) on [0,1] = %.5f +/- %.5f | true %.5f" %
      (I_est, se, I_true))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: number of samples N, and the random seed
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: sample size N (pi estimate error) ===")
Ns = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
errs = []
for n in Ns:
    e = []
    for seed in range(20):                       # 20 repeated experiments per N
        r = np.random.default_rng(seed)
        a, b = r.random(n), r.random(n)
        e.append(abs(4 * np.mean(a ** 2 + b ** 2 <= 1) - np.pi))
    errs.append(np.mean(e))
    print("N = %8d | mean |error| over 20 seeds = %.5f | 1/sqrt(N) = %.5f" %
          (n, errs[-1], 1 / np.sqrt(n)))
seed_est = []
for s in (1, 2, 3, 4):
    r = np.random.default_rng(s)
    a, b = r.random(1000), r.random(1000)
    seed_est.append(round(float(4 * np.mean(a ** 2 + b ** 2 <= 1)), 3))
print("different seeds, N = 1000:", seed_est)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
sel = slice(0, 2000)
axs[0].scatter(px[sel][inside[sel]], py[sel][inside[sel]], s=3, color="tab:blue")
axs[0].scatter(px[sel][~inside[sel]], py[sel][~inside[sel]], s=3, color="tab:red")
axs[0].set_aspect("equal")
axs[0].set_title("Random points: blue inside quarter circle")
axs[1].loglog(Ns, errs, "o-", label="mean error")
axs[1].loglog(Ns, 1 / np.sqrt(Ns), "k--", label="1/sqrt(N)")
axs[1].set_xlabel("N")
axs[1].set_ylabel("Error in pi estimate")
axs[1].set_title("Monte Carlo error shrinks like 1/sqrt(N)")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task20_pi.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: project schedule risk
#    3 sequential tasks, durations (min, most likely, max) in days
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: project schedule risk, 100000 simulations ===")
rng = np.random.default_rng(2026)
n = 100_000
design = rng.triangular(5, 7, 12, n)
build = rng.triangular(10, 15, 25, n)
test = rng.triangular(4, 6, 10, n)


def summarise(total, deadline):
    return (total.mean(), np.percentile(total, 95), np.mean(total <= deadline))


total = design + build + test
m, p95, prob = summarise(total, 35)
print("mean duration  = %.2f days" % m)
print("95th percentile = %.2f days" % p95)
print("P(finish within 35 days) = %.3f" % prob)
print("P(finish within 40 days) = %.3f" % np.mean(total <= 40))
print("simple plan (sum of most likely) = %d days" % (7 + 15 + 6))
print("P(finish within the simple plan of 28 days) = %.3f" % np.mean(total <= 28))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(total, bins=60, color="tab:blue", alpha=0.75)
ax.axvline(35, color="r", ls="--", label="deadline 35 days")
ax.axvline(p95, color="k", ls=":", label="95th percentile %.1f" % p95)
ax.set_xlabel("Total project duration (days)")
ax.set_ylabel("Frequency")
ax.set_title("Task 20 (new case): simulated project duration")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task20_project.png", dpi=150)
plt.close(fig)
