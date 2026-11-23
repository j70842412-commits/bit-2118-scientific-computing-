"""
BIT 2118 Scientific Computing - Task 18
Numerical Solution of ODEs Using SciPy (solve_ivp): SIR epidemic and spring-mass
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def sir(t, y, beta, gamma, N):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


def run_sir(beta, gamma=0.1, N=1000.0, I0=1.0, days=160):
    t_eval = np.linspace(0, days, 801)
    sol = solve_ivp(sir, (0, days), [N - I0, I0, 0.0], args=(beta, gamma, N),
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol


# ------------------------------------------------------------------
# A. MAIN CASE: SIR epidemic model, beta = 0.3, gamma = 0.1 (R0 = 3)
# ------------------------------------------------------------------
print("=== A. MAIN CASE: SIR model, N = 1000, beta = 0.3, gamma = 0.1 ===")
sol = run_sir(0.3)
S, I, R = sol.y
print("solver status:", sol.message)
print("R0 = beta/gamma = %.1f" % (0.3 / 0.1))
print("peak infected  = %.1f people on day %.1f" % (I.max(), sol.t[I.argmax()]))
print("final recovered = %.1f (%.1f %% of population)" % (R[-1], 100 * R[-1] / 1000))
print("never infected  = %.1f" % S[-1])

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: beta 0.3 -> 0.15 (e.g. contact reduction), R0 = 1.5
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: beta = 0.15 (R0 = 1.5) ===")
sol2 = run_sir(0.15)
S2, I2, R2 = sol2.y
print("peak infected  = %.1f people on day %.1f" % (I2.max(), sol2.t[I2.argmax()]))
print("final recovered = %.1f (%.1f %% of population)" % (R2[-1], 100 * R2[-1] / 1000))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, so, title in [(axs[0], sol, "beta = 0.3 (R0 = 3)"),
                      (axs[1], sol2, "beta = 0.15 (R0 = 1.5)")]:
    ax.plot(so.t, so.y[0], label="Susceptible")
    ax.plot(so.t, so.y[1], label="Infected")
    ax.plot(so.t, so.y[2], label="Recovered")
    ax.set_xlabel("Day")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("People")
axs[0].legend()
fig.suptitle("Task 18: SIR model solved with solve_ivp")
fig.tight_layout()
fig.savefig(FIG / "task18_sir.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: damped spring-mass (vehicle suspension)
#    m x'' + c x' + k x = 0  ->  system  x' = v,  v' = -(c v + k x)/m
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: suspension, m = 1, k = 4, initial x = 1 m ===")


def spring(t, y, m, c, k):
    x, v = y
    return [v, -(c * v + k * x) / m]


fig, ax = plt.subplots(figsize=(7, 3.8))
t_eval = np.linspace(0, 30, 1200)
for c, label in [(0.5, "underdamped c = 0.5"), (4.0, "critically damped c = 4"),
                 (8.0, "overdamped c = 8")]:
    s = solve_ivp(spring, (0, 30), [1.0, 0.0], args=(1.0, c, 4.0),
                  t_eval=t_eval, rtol=1e-8, atol=1e-10)
    x = s.y[0]
    big = x[np.abs(x) > 1e-3]                    # ignore numerical noise near zero
    cross = np.sum(np.diff(np.sign(big)) != 0)
    settle = t_eval[np.where(np.abs(x) > 0.02)[0].max()] if np.any(np.abs(x) > 0.02) else 0
    print("%-26s | zero crossings %d | settles within 2 %% after %.2f s" %
          (label, cross, settle))
    ax.plot(t_eval, x, label=label)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement x (m)")
ax.set_title("Task 18 (new case): damped spring-mass")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task18_spring.png", dpi=150)
plt.close(fig)
