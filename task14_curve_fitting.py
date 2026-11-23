"""
BIT 2118 Scientific Computing - Task 14
Curve Fitting and Approximation (scipy.optimize.curve_fit, np.polyfit)
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def r_squared(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def make_data(noise_sd, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 21, dtype=float)
    return t, logistic(t, 1000, 0.5, 10) + rng.normal(0, noise_sd, t.size)


# ------------------------------------------------------------------
# A. MAIN CASE: growth curve of a population (true K=1000, r=0.5, t0=10)
# ------------------------------------------------------------------
print("=== A. MAIN CASE: logistic growth curve, noise sd = 20 ===")
t, y = make_data(20)
popt, pcov = curve_fit(logistic, t, y, p0=[800, 0.3, 8])
perr = np.sqrt(np.diag(pcov))
print("fitted K  = %.2f +/- %.2f   (true 1000)" % (popt[0], perr[0]))
print("fitted r  = %.4f +/- %.4f (true 0.5)" % (popt[1], perr[1]))
print("fitted t0 = %.3f +/- %.3f  (true 10)" % (popt[2], perr[2]))
print("R^2 logistic = %.5f" % r_squared(y, logistic(t, *popt)))
poly = np.polyfit(t, y, 3)
print("R^2 cubic polynomial = %.5f" % r_squared(y, np.polyval(poly, t)))
print("prediction at t = 25: logistic %.1f | cubic polynomial %.1f" %
      (logistic(25, *popt), np.polyval(poly, 25)))

tt = np.linspace(0, 25, 300)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(t, y, color="k", s=18, label="data")
ax.plot(tt, logistic(tt, *popt), label="logistic fit")
ax.plot(tt, np.polyval(poly, tt), "--", label="cubic polynomial")
ax.set_ylim(-100, 1300)
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.set_title("Task 14: curve fitting a growth curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task14_logistic.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: measurement noise 20 -> 100
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: noise sd 20 -> 100 ===")
t2, y2 = make_data(100)
popt2, pcov2 = curve_fit(logistic, t2, y2, p0=[800, 0.3, 8], maxfev=10000)
perr2 = np.sqrt(np.diag(pcov2))
print("fitted K  = %.2f +/- %.2f" % (popt2[0], perr2[0]))
print("fitted r  = %.4f +/- %.4f" % (popt2[1], perr2[1]))
print("fitted t0 = %.3f +/- %.3f" % (popt2[2], perr2[2]))
print("R^2 = %.5f" % r_squared(y2, logistic(t2, *popt2)))
print("uncertainty in K grew by a factor of %.1f" % (perr2[0] / perr[0]))

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: demand forecasting (monthly mobile payments)
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: demand forecast, 12 months of simulated data ===")
rng = np.random.default_rng(7)
m = np.arange(1, 13, dtype=float)
demand = 200 * np.exp(0.06 * m) + rng.normal(0, 8, m.size)   # KES millions
expo = lambda x, a, b: a * np.exp(b * x)
pe, _ = curve_fit(expo, m, demand, p0=[150, 0.05])
pl = np.polyfit(m, demand, 1)
print("exponential fit: a = %.2f, b = %.4f (growth %.2f %% per month)" %
      (pe[0], pe[1], 100 * (np.exp(pe[1]) - 1)))
print("linear fit     : slope %.2f, intercept %.2f" % (pl[0], pl[1]))
print("R^2 exponential = %.4f | R^2 linear = %.4f" %
      (r_squared(demand, expo(m, *pe)), r_squared(demand, np.polyval(pl, m))))
print("forecast month 15: exponential %.1f | linear %.1f (KES million)" %
      (expo(15, *pe), np.polyval(pl, 15)))

mm = np.linspace(1, 15, 200)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(m, demand, color="k", s=18, label="data")
ax.plot(mm, expo(mm, *pe), label="exponential fit")
ax.plot(mm, np.polyval(pl, mm), "--", label="linear fit")
ax.set_xlabel("Month")
ax.set_ylabel("Demand (KES million)")
ax.set_title("Task 14 (new case): demand forecast")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task14_demand.png", dpi=150)
plt.close(fig)
