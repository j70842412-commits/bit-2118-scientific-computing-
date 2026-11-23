"""
BIT 2118 Scientific Computing - Task 12
Polynomial and Nonlinear Equation Solving (np.roots, scipy.optimize.root)
"""
import numpy as np
from scipy.optimize import root, root_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=5, suppress=True)

# ------------------------------------------------------------------
# A. MAIN CASE
#    1) polynomial roots   x^3 - 6x^2 + 11x - 6 = 0
#    2) kinematics: projectile height  h(t) = -4.9 t^2 + v0 t + h0 = 0
#    3) geometric intersection of a circle and a parabola (nonlinear system)
# ------------------------------------------------------------------
print("=== A. MAIN CASE ===")
r = np.roots([1, -6, 11, -6])
print("roots of x^3 - 6x^2 + 11x - 6:", np.sort(r.real))


def flight_time(v0, h0=1.5):
    rt = np.roots([-4.9, v0, h0])
    return rt[rt.real > 0].real.max()


print("projectile (v0 = 20 m/s, h0 = 1.5 m) lands after %.4f s" % flight_time(20))


def system(p):
    x, y = p
    return [x ** 2 + y ** 2 - 25, y - (x ** 2 - 5)]


print("circle x^2+y^2 = 25 and parabola y = x^2 - 5:")
found = []
for guess in [(3, 4), (-3, 4), (0.5, -4)]:
    sol = root(system, guess)
    found.append(sol.x)
    print("  guess %-10s -> (x, y) = (%.5f, %.5f), success=%s" %
          (guess, sol.x[0], sol.x[1], sol.success))

th = np.linspace(0, 2 * np.pi, 400)
xs = np.linspace(-4, 4, 400)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(5 * np.cos(th), 5 * np.sin(th), label="circle")
ax.plot(xs, xs ** 2 - 5, label="parabola")
fx = np.array(found)
ax.plot(fx[:, 0], fx[:, 1], "ro", label="intersections")
ax.set_aspect("equal")
ax.set_ylim(-6, 6)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Task 12: nonlinear system, three solutions")
fig.tight_layout()
fig.savefig(FIG / "task12_intersection.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: launch speed 20 -> 30 m/s, and the starting guess
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE ===")
for v0 in [20, 30]:
    print("v0 = %d m/s -> flight time %.4f s" % (v0, flight_time(v0)))
for guess in [(2.5, 3.5), (-2.5, 3.5), (0.1, -4.9), (4, 4)]:
    sol = root(system, guess)
    print("start %-10s converges to (%.4f, %.4f)" % (guess, sol.x[0], sol.x[1]))

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: chemical equilibrium A + B <-> C
#    Kc = x / ((a - x)(b - x)), initial [A] = [B] = 1.0 mol/L
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: chemical equilibrium, A + B <-> C ===")
a = b = 1.0
fig, ax = plt.subplots(figsize=(7, 3.8))
xx = np.linspace(0, 0.99, 400)
for Kc in [4.0, 10.0]:
    # Kc (a-x)(b-x) - x = 0  ->  Kc x^2 - (Kc(a+b)+1) x + Kc a b = 0
    rts = np.roots([Kc, -(Kc * (a + b) + 1), Kc * a * b])
    x_eq = rts[(rts.real > 0) & (rts.real < min(a, b))].real[0]
    chk = root_scalar(lambda x: x / ((a - x) * (b - x)) - Kc, bracket=[0, 0.99]).root
    print("Kc = %4.1f -> roots %s | feasible x = %.5f (root_scalar %.5f)" %
          (Kc, np.round(rts.real, 4), x_eq, chk))
    print("          [C] = %.4f, [A] = [B] = %.4f mol/L" % (x_eq, a - x_eq))
    ax.plot(xx, xx / ((a - xx) * (b - xx)), label="Kc = %.0f curve" % Kc)
    ax.axhline(Kc, ls=":", color="gray")
    ax.plot(x_eq, Kc, "ro")
ax.set_ylim(0, 25)
ax.set_xlabel("Amount reacted x (mol/L)")
ax.set_ylabel("Reaction quotient")
ax.set_title("Task 12 (new case): equilibrium where quotient = Kc")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task12_equilibrium.png", dpi=150)
plt.close(fig)
