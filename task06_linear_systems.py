"""
BIT 2118 Scientific Computing - Task 6
Solution of Systems of Linear Equations (circuit analysis, resource allocation)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=5, suppress=True)


def mesh_currents(R1, R2, R3, R4, R5, V1, V2):
    """Three-loop resistor circuit solved by mesh analysis: A I = b."""
    A = np.array([[R1 + R2, -R2, 0.0],
                  [-R2, R2 + R3 + R4, -R4],
                  [0.0, -R4, R4 + R5]])
    b = np.array([V1, 0.0, -V2])
    return A, b, np.linalg.solve(A, b)


# ------------------------------------------------------------------
# A. MAIN CASE: three-loop circuit
# ------------------------------------------------------------------
print("=== A. MAIN CASE: mesh-current circuit ===")
A, b, I = mesh_currents(10, 20, 30, 40, 50, 12, 6)
print("A (ohms) =\n", A)
print("b (volts) =", b)
print("mesh currents I1, I2, I3 (A) =", I)
print("residual ||A I - b|| = %.2e" % np.linalg.norm(A @ I - b))
print("condition number of A = %.2f" % np.linalg.cond(A))
print("current through R2 (I1-I2) = %.5f A" % (I[0] - I[1]))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: R2 from 20 ohm to 40 ohm
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: R2 20 -> 40 ohm ===")
_, _, I2 = mesh_currents(10, 40, 30, 40, 50, 12, 6)
print("new currents (A)  =", I2)
print("change (A)        =", I2 - I)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
idx = np.arange(3)
ax.bar(idx - 0.2, I * 1000, 0.4, label="R2 = 20 ohm")
ax.bar(idx + 0.2, I2 * 1000, 0.4, label="R2 = 40 ohm")
ax.set_xticks(idx)
ax.set_xticklabels(["I1", "I2", "I3"])
ax.set_ylabel("Current (mA)")
ax.set_title("Task 6: effect of changing R2 on mesh currents")
ax.axhline(0, color="k", lw=0.8)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task06_currents.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: farm resource allocation (3 crops)
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: allocating land, water and labour to 3 crops ===")
#            maize beans vegetables
M = np.array([[1.0, 1.0, 1.0],       # land used (ha per ha)  -> 100 ha available
              [3.0, 2.0, 4.0],       # water (ML per ha)      -> 300 ML available
              [2.0, 1.0, 2.0]])      # labour (people per ha) -> 170 people available
rhs = np.array([100.0, 300.0, 170.0])
x = np.linalg.solve(M, rhs)
print("det(M) = %.2f (non-zero so a unique solution exists)" % np.linalg.det(M))
print("hectares: maize %.2f, beans %.2f, vegetables %.2f" % tuple(x))
print("check M @ x =", M @ x)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.bar(["Maize", "Beans", "Vegetables"], x, color=["gold", "tab:brown", "tab:green"])
ax.set_ylabel("Hectares")
ax.set_title("Task 6 (new case): land allocation")
fig.tight_layout()
fig.savefig(FIG / "task06_farm.png", dpi=150)
plt.close(fig)
