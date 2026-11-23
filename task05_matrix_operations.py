"""
BIT 2118 Scientific Computing - Task 5
Matrix Algebra and Matrix Operations (computer-graphics transformation)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# unit square, columns are the corner points (first corner repeated to close it)
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]], dtype=float)

# ------------------------------------------------------------------
# A. MAIN CASE: scale by (2, 1) then rotate by 45 degrees
# ------------------------------------------------------------------
print("=== A. MAIN CASE: T = R(45) @ S(2,1) ===")
S = np.diag([2.0, 1.0])
R45 = rot(45)
T = R45 @ S                                  # matrix product
print("S =\n", S)
print("R(45) =\n", R45)
print("T = R @ S =\n", T)
print("T transpose =\n", T.T)
print("det(T) = %.4f  (area scale factor)" % np.linalg.det(T))
Tinv = np.linalg.inv(T)
print("inverse of T =\n", Tinv)
print("T @ inv(T) =\n", T @ Tinv)
new_pts = T @ square
print("transformed corners (x row, y row) =\n", new_pts[:, :4])

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: rotation angle 45 -> 90 and 30 degrees
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: rotation angle ===")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(square[0], square[1], "k-o", label="original")
for ang, c in [(45, "tab:blue"), (90, "tab:red"), (30, "tab:green")]:
    Tn = rot(ang) @ S
    p = Tn @ square
    print("angle %2d deg: det = %.3f, corner (1,1) -> (%.3f, %.3f)" %
          (ang, np.linalg.det(Tn), p[0, 2], p[1, 2]))
    ax.plot(p[0], p[1], "-o", color=c, label="scale then rotate %d deg" % ang)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.set_title("Task 5: matrix transformation of a square")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "task05_transform.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: bakery ingredient requirements and cost
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: bakery planning with matrix products ===")
products = ["bread", "cake", "mandazi"]
ingredients = ["flour kg", "sugar kg", "oil L"]
Q = np.array([[0.50, 0.02, 0.05],            # ingredient per unit of each product
              [0.30, 0.25, 0.10],
              [0.25, 0.10, 0.15]])
price = np.array([70.0, 150.0, 350.0])       # assumed KES per ingredient unit
plan = np.array([200, 50, 300])              # units to bake today
need = plan @ Q                              # total ingredient needed
unit_cost = Q @ price                        # cost of one unit of each product
total_cost = plan @ unit_cost
print("ingredients needed :", {k: float(v) for k, v in zip(ingredients, need)})
print("unit cost (KES)    :", {k: float(v) for k, v in zip(products, unit_cost)})
print("total cost (KES)   : %.2f" % total_cost)
print("check plan@Q@price : %.2f" % (plan @ Q @ price))

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(products, plan * unit_cost, color=["tab:orange", "tab:pink", "tab:brown"])
ax.set_ylabel("Ingredient cost (KES)")
ax.set_title("Task 5 (new case): ingredient cost by product")
fig.tight_layout()
fig.savefig(FIG / "task05_bakery.png", dpi=150)
plt.close(fig)
