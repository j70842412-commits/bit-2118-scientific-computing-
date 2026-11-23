"""
BIT 2118 Scientific Computing - Task 17
Ordinary Differential Equations: Euler Method (population growth, RC circuit)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def euler(f, y0, t_end, h):
    n = int(round(t_end / h))
    t = np.linspace(0, n * h, n + 1)
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])          # y_{n+1} = y_n + h f(t_n, y_n)
    return t, y


# ------------------------------------------------------------------
# A. MAIN CASE: logistic population growth  dP/dt = r P (1 - P/K)
# ------------------------------------------------------------------
r, K, P0 = 0.4, 1000.0, 50.0
f_pop = lambda t, P: r * P * (1 - P / K)
exact_pop = lambda t: K / (1 + (K - P0) / P0 * np.exp(-r * t))

print("=== A. MAIN CASE: logistic growth, r = 0.4/day, K = 1000, P0 = 50 ===")
t1, P1 = euler(f_pop, P0, 30, 1.0)
print("Euler h = 1: P(30) = %.3f | exact %.3f | error %.3f" %
      (P1[-1], exact_pop(30), abs(P1[-1] - exact_pop(30))))
print("first five Euler values:", np.round(P1[:5], 3))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: step size h = 5, 1, 0.1
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: step size h ===")
fig, ax = plt.subplots(figsize=(7, 3.8))
tf = np.linspace(0, 30, 300)
ax.plot(tf, exact_pop(tf), "k-", lw=2, label="exact")
for h in [5.0, 1.0, 0.1]:
    t, P = euler(f_pop, P0, 30, h)
    max_err = np.abs(P - exact_pop(t)).max()
    print("h = %-4g | P(30) = %9.3f | max error %.3f" % (h, P[-1], max_err))
    ax.plot(t, P, "--", label="Euler h = %g" % h)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Population")
ax.set_title("Task 17: Euler method, logistic growth")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task17_population.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: charging an RC circuit (tau = RC = 1 s)
#    dVc/dt = (Vs - Vc)/RC,  R = 10 kOhm, C = 100 uF, Vs = 5 V
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: capacitor charging, tau = 1 s, Vs = 5 V ===")
Vs, tau = 5.0, 10e3 * 100e-6
f_rc = lambda t, V: (Vs - V) / tau
exact_rc = lambda t: Vs * (1 - np.exp(-t / tau))
fig, ax = plt.subplots(figsize=(7, 3.8))
tf = np.linspace(0, 6, 300)
ax.plot(tf, exact_rc(tf), "k-", lw=2, label="exact")
for h in [0.1, 0.5, 1.0, 2.5]:
    t, V = euler(f_rc, 0.0, 6.0, h)
    err = np.abs(V - exact_rc(t)).max()
    status = "stable" if h < 2 * tau else "UNSTABLE (h > 2 tau)"
    print("h = %-3g | V(%.1f s) = %9.4f V | exact %.4f V | max error %.4f | %s" %
          (h, t[-1], V[-1], exact_rc(t[-1]), err, status))
    if h in (0.1, 0.5, 2.5):
        ax.plot(t, V, "o--", ms=3, label="Euler h = %g s" % h)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Capacitor voltage (V)")
ax.set_title("Task 17 (new case): RC charging, Euler method")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task17_rc.png", dpi=150)
plt.close(fig)
