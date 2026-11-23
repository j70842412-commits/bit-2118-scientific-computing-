"""
BIT 2118 Scientific Computing - Task 16
Numerical Integration (trapezoidal rule and Simpson's rule)
"""
import numpy as np
from scipy.integrate import simpson, trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

T_END = 10.0


def velocity(t):
    return 10 * (1 - np.exp(-0.3 * t))          # m/s


exact = 10 * (T_END - (1 - np.exp(-0.3 * T_END)) / 0.3)      # exact distance

# ------------------------------------------------------------------
# A. MAIN CASE: distance travelled from velocity data
# ------------------------------------------------------------------
print("=== A. MAIN CASE: distance = integral of v(t) dt, 0..10 s ===")
t = np.linspace(0, T_END, 11)
v = velocity(t)
d_trap = trapezoid(v, t)
d_simp = simpson(v, x=t)
print("exact distance        = %.6f m" % exact)
print("trapezoidal (11 pts)  = %.6f m (error %.2e)" % (d_trap, abs(d_trap - exact)))
print("Simpson     (11 pts)  = %.6f m (error %.2e)" % (d_simp, abs(d_simp - exact)))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: number of sample points
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: number of sample points ===")
Ns = [3, 5, 11, 21, 41, 101]
e_t, e_s = [], []
for N in Ns:
    tt = np.linspace(0, T_END, N)
    e_t.append(abs(trapezoid(velocity(tt), tt) - exact))
    e_s.append(abs(simpson(velocity(tt), x=tt) - exact) + 1e-16)
    print("N = %3d | trapezoid error %.3e | Simpson error %.3e" % (N, e_t[-1], e_s[-1]))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
tf = np.linspace(0, T_END, 300)
axs[0].plot(tf, velocity(tf), "k-")
axs[0].fill_between(tf, velocity(tf), alpha=0.25)
axs[0].plot(t, v, "ro")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Velocity (m/s)")
axs[0].set_title("Area under v(t) = distance (%.2f m)" % exact)
axs[0].grid(alpha=0.3)
axs[1].loglog(Ns, e_t, "o-", label="trapezoid")
axs[1].loglog(Ns, e_s, "s-", label="Simpson")
axs[1].set_xlabel("Number of points N")
axs[1].set_ylabel("Absolute error (m)")
axs[1].set_title("Accuracy vs N")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task16_distance.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: daily household energy from a power profile
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: energy consumption from hourly power readings ===")
h = np.arange(0, 25, dtype=float)                                  # 25 readings
P = 0.5 + 1.2 * np.exp(-((h - 7) / 1.5) ** 2) + 2.0 * np.exp(-((h - 19) / 2.0) ** 2)
tariff = 25.0                                                      # assumed KES/kWh
E_trap = trapezoid(P, h)
E_simp = simpson(P, x=h)
hf = np.linspace(0, 24, 20001)
Pf = 0.5 + 1.2 * np.exp(-((hf - 7) / 1.5) ** 2) + 2.0 * np.exp(-((hf - 19) / 2.0) ** 2)
E_ref = trapezoid(Pf, hf)                                          # fine reference
print("reference energy (20001 pts) = %.4f kWh" % E_ref)
print("trapezoid (hourly)           = %.4f kWh" % E_trap)
print("Simpson   (hourly)           = %.4f kWh" % E_simp)
P3, h3 = P[::3], h[::3]
print("Simpson   (every 3 hours)    = %.4f kWh" % simpson(P3, x=h3))
print("daily cost at KES %.0f/kWh    = KES %.2f" % (tariff, E_simp * tariff))
print("peak demand %.2f kW at hour %d" % (P.max(), h[P.argmax()]))

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.fill_between(h, P, alpha=0.3)
ax.plot(h, P, "o-")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Power (kW)")
ax.set_title("Task 16 (new case): energy = area under power curve (%.1f kWh)" % E_simp)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task16_energy.png", dpi=150)
plt.close(fig)
