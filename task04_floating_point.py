"""
BIT 2118 Scientific Computing - Task 4
Floating-Point Arithmetic and Numerical Errors
"""
import math
from decimal import Decimal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# A. MAIN CASE: representation error, accumulation, cancellation
# ------------------------------------------------------------------
print("=== A. MAIN CASE ===")
print("0.1 + 0.2             =", repr(0.1 + 0.2))
print("0.1 + 0.2 == 0.3     :", 0.1 + 0.2 == 0.3)
print("abs error            : %.3e" % abs(0.1 + 0.2 - 0.3))
print("np.isclose(0.1+0.2, 0.3):", np.isclose(0.1 + 0.2, 0.3))
print("machine epsilon float64: %.3e" % np.finfo(np.float64).eps)
print("machine epsilon float32: %.3e" % np.finfo(np.float32).eps)

n = 1_000_000
s = 0.0
for _ in range(n):
    s += 0.1
print("\nadd 0.1 one million times:")
print("  naive loop sum     = %.10f (error %.3e)" % (s, abs(s - 100000)))
print("  math.fsum          = %.10f" % math.fsum([0.1] * n))

# catastrophic cancellation: f(x) = (1 - cos x) / x^2, true limit 0.5
x = np.logspace(-9, 0, 300)
naive = (1 - np.cos(x)) / x ** 2
stable = 2 * np.sin(x / 2) ** 2 / x ** 2             # algebraically identical
rel_err = np.abs(naive - stable) / stable
print("\n(1-cos x)/x^2 at x = 1e-8:")
print("  naive formula      =", (1 - np.cos(1e-8)) / 1e-8 ** 2, " (true value 0.5)")
print("  stable formula     =", 2 * np.sin(1e-8 / 2) ** 2 / 1e-8 ** 2)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.loglog(x, np.maximum(rel_err, 1e-17), color="tab:red")
ax.set_xlabel("x")
ax.set_ylabel("Relative error of naive formula")
ax.set_title("Task 4: cancellation error in (1 - cos x)/x^2")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task04_cancellation.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: precision float64 -> float32
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: float64 -> float32 ===")
s32 = np.float32(0.0)
inc = np.float32(0.1)
for _ in range(n):
    s32 += inc
print("float64 sum of 1e6 x 0.1 = %.6f (error %.3e)" % (s, abs(s - 100000)))
print("float32 sum of 1e6 x 0.1 = %.6f (error %.3e)" %
      (float(s32), abs(float(s32) - 100000)))

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: banking, adding KES 0.10 service fees
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: 1,000,000 fees of KES 0.10 in a bank ledger ===")
fee = 0.10
total_f64 = 0.0
total_dec = Decimal("0.00")
fee_dec = Decimal("0.10")
for _ in range(n):
    total_f64 += fee
    total_dec += fee_dec
print("float64 ledger total : %.8f" % total_f64)
print("float32 ledger total : %.8f" % float(s32))
print("Decimal ledger total : %s  (exact)" % total_dec)
print("float64 error in cents: %.6f" % (abs(total_f64 - float(total_dec)) * 100))
print("float32 error in KES  : %.4f" % abs(float(s32) - float(total_dec)))
print("rounded to 2 dp float64: %.2f | Decimal: %s" %
      (round(total_f64, 2), total_dec.quantize(Decimal("0.01"))))
