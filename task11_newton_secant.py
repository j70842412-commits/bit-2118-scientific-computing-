"""
BIT 2118 Scientific Computing - Task 11
Root Finding: Newton-Raphson and Secant Methods (diode circuit, bond yield)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)


def newton(f, df, x0, tol=1e-12, maxit=50):
    x = x0
    hist = [x]
    for k in range(1, maxit + 1):
        x_new = x - f(x) / df(x)
        hist.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, k, hist
        x = x_new
    return x, maxit, hist


def secant(f, x0, x1, tol=1e-12, maxit=50):
    hist = [x0, x1]
    for k in range(1, maxit + 1):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        hist.append(x2)
        if abs(x2 - x1) < tol:
            return x2, k, hist
        x0, x1 = x1, x2
    return x1, maxit, hist


# ------------------------------------------------------------------
# A. MAIN CASE: diode circuit. Solve (Vs - V)/R = Is (exp(V/Vt) - 1) for V.
# ------------------------------------------------------------------
Is, Vt = 1e-12, 0.02585


def diode(Vs, R):
    f = lambda V: (Vs - V) / R - Is * (np.exp(V / Vt) - 1)
    df = lambda V: -1 / R - Is / Vt * np.exp(V / Vt)
    return f, df


print("=== A. MAIN CASE: diode voltage, Vs = 5 V, R = 1 kOhm ===")
f, df = diode(5.0, 1000.0)
Vn, kn, hn = newton(f, df, 0.6)
Vs_, ks, hs = secant(f, 0.5, 0.6)
print("Newton-Raphson: V = %.8f V in %d iterations" % (Vn, kn))
print("Secant        : V = %.8f V in %d iterations" % (Vs_, ks))
print("diode current   I = %.4f mA" % ((5.0 - Vn) / 1000 * 1e3))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: Vs = 9 V and R = 470 ohm
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: Vs = 9 V, R = 470 ohm ===")
f2, df2 = diode(9.0, 470.0)
Vn2, kn2, _ = newton(f2, df2, 0.6)
Vs2, ks2, _ = secant(f2, 0.5, 0.6)
print("Newton-Raphson: V = %.8f V in %d iterations" % (Vn2, kn2))
print("Secant        : V = %.8f V in %d iterations" % (Vs2, ks2))
print("diode current   I = %.4f mA" % ((9.0 - Vn2) / 470 * 1e3))
print("a larger supply raised the diode voltage by only %.4f V" % (Vn2 - Vn))

en = np.abs(np.array(hn) - Vn) + 1e-17
es = np.abs(np.array(hs) - Vn) + 1e-17
V = np.linspace(0.4, 0.7, 300)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(V, f(V) * 1e3)
axs[0].axhline(0, color="k", lw=0.8)
axs[0].plot(Vn, 0, "ro")
axs[0].set_xlabel("Diode voltage V (V)")
axs[0].set_ylabel("f(V) (mA)")
axs[0].set_title("Diode equation f(V) = 0")
axs[0].grid(alpha=0.3)
axs[1].semilogy(en, "o-", label="Newton-Raphson")
axs[1].semilogy(es, "s-", label="Secant")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("|error| (V)")
axs[1].set_title("Convergence")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task11_diode.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: yield to maturity of a 10-year bond
#    face 100, coupon 12 per year, market price 95
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: bond yield to maturity (finance) ===")
face, coupon, years, price = 100.0, 12.0, 10, 95.0
t = np.arange(1, years + 1)


def pv(y):
    return np.sum(coupon / (1 + y) ** t) + face / (1 + y) ** years


fy = lambda y: pv(y) - price
dfy = lambda y, h=1e-6: (fy(y + h) - fy(y - h)) / (2 * h)       # numerical derivative
yn, kyn, hyn = newton(fy, dfy, 0.10, tol=1e-12)
ys, kys, hys = secant(fy, 0.08, 0.12, tol=1e-12)
print("Newton-Raphson: yield = %.6f %% in %d iterations" % (100 * yn, kyn))
print("Secant        : yield = %.6f %% in %d iterations" % (100 * ys, kys))
print("check: price at that yield = %.6f" % pv(yn))
print("coupon rate is 12 %% but price is below face, so yield (%.2f %%) > 12 %%" %
      (100 * yn))

yy = np.linspace(0.05, 0.2, 200)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(yy * 100, [pv(y) for y in yy])
ax.axhline(price, color="r", ls="--", label="market price 95")
ax.plot(yn * 100, price, "ko", label="yield %.2f %%" % (100 * yn))
ax.set_xlabel("Yield (%)")
ax.set_ylabel("Bond price")
ax.set_title("Task 11 (new case): price-yield curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task11_bond.png", dpi=150)
plt.close(fig)
