"""
BIT 2118 Scientific Computing - Task 19
Numerical Optimization (scipy.optimize.minimize): cloud cost, ML training
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# A. MAIN CASE: cloud resource allocation
#    x = small servers (capacity 1, cost 0.05 units per hour)
#    y = large servers (capacity 4, cost 0.16 units per hour, at most 20 allowed)
#    cost = 0.05 x + 0.16 y + 400/(x + 4y)   (last term = slowness penalty)
#    constraint: total capacity x + 4y >= demand
# ------------------------------------------------------------------


def cost(p):
    x, y = p
    return 0.05 * x + 0.16 * y + 400.0 / (x + 4 * y)


def optimise(demand):
    cons = [{"type": "ineq", "fun": lambda p: p[0] + 4 * p[1] - demand}]
    bnds = [(0, None), (0, 20)]
    return minimize(cost, x0=[10.0, 5.0], method="SLSQP", bounds=bnds, constraints=cons)


print("=== A. MAIN CASE: cloud allocation, demand >= 60 capacity units ===")
res = optimise(60)
x, y = res.x
print("success:", res.success, "| iterations:", res.nit)
print("small servers x = %.3f | large servers y = %.3f" % (x, y))
print("total capacity = %.3f | minimum cost = %.4f per hour" % (x + 4 * y, res.fun))

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: demand 60 -> 120
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: demand 60 -> 120 ===")
res2 = optimise(120)
x2, y2 = res2.x
print("small servers x = %.3f | large servers y = %.3f" % (x2, y2))
print("total capacity = %.3f | minimum cost = %.4f per hour" % (x2 + 4 * y2, res2.fun))
print("the large-server limit of 20 is active, so extra demand is met by small servers")

xs = np.linspace(0, 60, 200)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(xs, [cost([v, 20]) for v in xs], label="cost with y = 20")
ax.plot(x, cost([x, y]), "ro", label="optimum, demand 60")
ax.plot(x2, cost([x2, y2]), "gs", label="constrained optimum, demand 120")
ax.set_xlabel("Number of small servers x")
ax.set_ylabel("Cost per hour")
ax.set_title("Task 19: cloud cost minimisation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "task19_cloud.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: machine-learning training by gradient descent
#    fit y = w x + b to 50 noisy points by minimising mean squared error
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: training a linear model (gradient descent) ===")
rng = np.random.default_rng(11)
xd = rng.uniform(0, 10, 50)
yd = 2.5 * xd + 4.0 + rng.normal(0, 2.0, 50)


def mse(p):
    return np.mean((p[0] * xd + p[1] - yd) ** 2)


def gradient_descent(lr, iters=400):
    w = b = 0.0
    losses = []
    for _ in range(iters):
        err = w * xd + b - yd
        w -= lr * 2 * np.mean(err * xd)
        b -= lr * 2 * np.mean(err)
        losses.append(np.mean((w * xd + b - yd) ** 2))
    return w, b, losses


w1, b1, loss1 = gradient_descent(0.01)
w2, b2, loss2 = gradient_descent(0.07, iters=60)
bfgs = minimize(mse, [0.0, 0.0], method="BFGS")
pf = np.polyfit(xd, yd, 1)
print("gradient descent (lr 0.01): w = %.4f, b = %.4f, final MSE %.4f" %
      (w1, b1, loss1[-1]))
print("scipy minimize (BFGS)     : w = %.4f, b = %.4f, MSE %.4f" %
      (bfgs.x[0], bfgs.x[1], bfgs.fun))
print("np.polyfit (closed form)  : w = %.4f, b = %.4f" % (pf[0], pf[1]))
print("learning rate 0.07 is too large: loss after 60 iterations = %.3e" % loss2[-1])

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].scatter(xd, yd, s=14, color="k", label="data")
xx = np.linspace(0, 10, 100)
axs[0].plot(xx, w1 * xx + b1, "r-", label="gradient descent fit")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].set_title("Fitted line")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[1].semilogy(loss1, label="lr = 0.01 (converges)")
axs[1].semilogy(loss2, label="lr = 0.07 (diverges)")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Mean squared error")
axs[1].set_title("Training loss")
axs[1].legend()
axs[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIG / "task19_training.png", dpi=150)
plt.close(fig)
