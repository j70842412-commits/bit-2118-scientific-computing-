"""
BIT 2118 Scientific Computing - Task 9
Eigenvalues and Eigenvectors (PageRank and Principal Component Analysis)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------------
# A. MAIN CASE: PageRank of a 5-page web
# ------------------------------------------------------------------
links = {0: [1, 2], 1: [2], 2: [0], 3: [0, 2, 4], 4: [3]}   # page -> pages it links to
n = 5
M = np.zeros((n, n))
for src, outs in links.items():
    for dst in outs:
        M[dst, src] = 1.0 / len(outs)                       # column-stochastic matrix


def pagerank(M, d=0.85):
    n = M.shape[0]
    G = d * M + (1 - d) / n * np.ones((n, n))                # Google matrix
    vals, vecs = np.linalg.eig(G)
    k = np.argmax(vals.real)                                 # dominant eigenvalue (= 1)
    v = np.abs(vecs[:, k].real)
    return vals[k].real, v / v.sum()


def power_iteration(M, d=0.85, iters=200):
    n = M.shape[0]
    G = d * M + (1 - d) / n * np.ones((n, n))
    r = np.ones(n) / n
    for _ in range(iters):
        r = G @ r
        r /= r.sum()
    return r


print("=== A. MAIN CASE: PageRank, damping d = 0.85 ===")
print("link matrix M =\n", M)
lam, rank = pagerank(M, 0.85)
print("dominant eigenvalue = %.6f" % lam)
print("PageRank (eig)      :", rank)
print("PageRank (power it.):", power_iteration(M, 0.85))
print("ranking (best first): pages", [int(i) for i in np.argsort(-rank)])

# ------------------------------------------------------------------
# C. PARAMETER CHANGE: damping factor d
# ------------------------------------------------------------------
print("\n=== C. PARAMETER CHANGE: damping factor ===")
ds = [0.5, 0.85, 0.99]
ranks = []
for d in ds:
    _, r = pagerank(M, d)
    ranks.append(r)
    print("d = %.2f -> %s" % (d, r))

fig, ax = plt.subplots(figsize=(7, 3.8))
w = 0.25
for i, (d, r) in enumerate(zip(ds, ranks)):
    ax.bar(np.arange(n) + (i - 1) * w, r, w, label="d = %.2f" % d)
ax.set_xticks(range(n))
ax.set_xticklabels(["P%d" % i for i in range(n)])
ax.set_ylabel("PageRank score")
ax.set_title("Task 9: PageRank for different damping factors")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task09_pagerank.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# E. RELATED REAL-WORLD CASE: PCA on student marks (programming vs maths)
# ------------------------------------------------------------------
print("\n=== E. NEW CASE: PCA of 60 students' marks (simulated data) ===")
rng = np.random.default_rng(1)
prog = np.clip(rng.normal(60, 12, 60), 0, 100)
maths = np.clip(0.7 * prog + 15 + rng.normal(0, 8, 60), 0, 100)
X = np.column_stack([prog, maths])
Xc = X - X.mean(axis=0)                                      # centre the data
C = np.cov(Xc, rowvar=False)                                 # covariance matrix
vals, vecs = np.linalg.eigh(C)                               # symmetric -> eigh
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
print("covariance matrix =\n", C)
print("eigenvalues (variances):", vals)
print("explained variance ratio:", vals / vals.sum())
print("PC1 direction           :", vecs[:, 0])
scores = Xc @ vecs[:, 0]
print("1-D scores of first 5 students on PC1:", scores[:5])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.7)
mu = X.mean(axis=0)
for i, c in enumerate(["tab:red", "tab:green"]):
    v = vecs[:, i] * 2 * np.sqrt(vals[i])
    ax.arrow(mu[0], mu[1], v[0], v[1], color=c, width=0.35, length_includes_head=True,
             label="PC%d (%.0f%% variance)" % (i + 1, 100 * vals[i] / vals.sum()))
ax.set_xlabel("Programming mark")
ax.set_ylabel("Mathematics mark")
ax.set_title("Task 9 (new case): principal components of marks")
ax.legend()
ax.axis("equal")
fig.tight_layout()
fig.savefig(FIG / "task09_pca.png", dpi=150)
plt.close(fig)
