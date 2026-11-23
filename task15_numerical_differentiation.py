"""
BIT 2118 Scientific Computing - Task 15
Numerical Differentiation (forward, central differences, np.gradient)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def position(t):
    return t ** 3 - 4 * t ** 2 + 6 * t          # metres


def velocity_true(t):
    return 3 * t ** 2 - 8 * t + 6               # m/s


def accel_true(t):
    return 6 * t - 8                            # m/s^2


def analyse(h):
    t = np.arange(0, 5 + 1e-9, h)
    s = position(t)
    v = np.gradient(s, t)                       # velocity from position data
    a = np.gradient(v, t)                       # acceleration from velocity
    inner = slice(2, -2)                        # ignore less accurate end points
    ev = np.abs(v - velocity_true(t))[inner].max()
    ea = np.abs(a - accel_true(t))[inner].max()
    return t, s, v, a, ev, ea


# ------------------------------------------------------------------
# A. MAIN CASE: speed and acceleration from position data, h = 0.5 s
# ------------------------------------------------------------------
print("=== A. MAIN CASE: position data sampled every h = 0.5 s ===")
t, s, v, a, ev, ea = analyse(0.5)
print("t (s)      :", t)
print("position m :", s)
print("velocity   :", v)
print("true vel.  :", velocity_true(t))
print("accel      :", a)
print("max interior error: velocity %.4f m/s, acceleration %.4f m/s^2" % (ev, ea))

# forward vs central difference on f(x) = sin(x) at x = 1
print("\nforward vs central difference for d/dx sin(x) at x = 1 (true cos 1 = %.6f)"
      % np.cos(1))
hs = np.array([0.5, 0.1, 0.01, 0.001])
fwd = np.abs((np.sin(1 + hs) - np.sin(1)) / hs - np.cos(1))
cen = np.abs((np.sin(1 + hs) - np.sin(1 - hs)) / (2 * hs) - np.cos(1))
for hh, e1, e2 in zip(hs, fwd, cen):
    print("h = %-6g forward error %.3e | central error %.3e" % (hh, e1, e2))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: step size h 0.5 -> 0.1
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: step size h ===")
for h in [0.5, 0.1]:
    *_, ev_, ea_ = analyse(h)
    print("h = %.1f | velocity error %.4f | acceleration error %.4f" % (h, ev_, ea_))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
tf = np.linspace(0, 5, 300)
axs[0].plot(tf, velocity_true(tf), "k-", label="true velocity")
axs[0].plot(t, v, "o--", label="np.gradient, h = 0.5")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Velocity (m/s)")
axs[0].set_title("Velocity from position data")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].loglog(hs, fwd, "o-", label="forward (order h)")
axs[1].loglog(hs, cen, "s-", label="central (order h^2)")
axs[1].set_xlabel("Step size h")
axs[1].set_ylabel("Error")
axs[1].set_title("Error vs step size, d/dx sin x at x=1")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task15_derivative.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: water-tank level -> net flow rate
#    tank area 20 m^2, level logged every 10 minutes
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: flow rate from tank level readings ===")
area = 20.0
tm = np.arange(0, 121, 10, dtype=float)                    # minutes
level = 3.0 - 0.8 * (1 - np.exp(-tm / 90))                 # metres (simulated)
flow_true = area * (0.8 / 90) * np.exp(-tm / 90) * -1      # m^3/min, negative = draining
flow = area * np.gradient(level, tm)
print("net flow estimate (m3/min):", flow)
print("true flow         (m3/min):", flow_true)
print("max interior error: %.5f m3/min" % np.abs(flow - flow_true)[1:-1].max())
rng = np.random.default_rng(5)
noisy = level + rng.normal(0, 0.005, level.size)           # 5 mm sensor noise
flow_noisy = area * np.gradient(noisy, tm)
print("with 5 mm sensor noise, max error becomes %.5f m3/min" %
      np.abs(flow_noisy - flow_true)[1:-1].max())

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(tm, flow_true, "k-", label="true flow")
ax.plot(tm, flow, "o--", label="from clean levels")
ax.plot(tm, flow_noisy, "s:", label="from noisy levels")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Net flow (m3/min)")
ax.set_title("Task 15 (new case): flow rate from level data")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task15_tank.png", dpi=150)
plt.close(fig)
