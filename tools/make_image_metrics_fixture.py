#!/usr/bin/env python3
"""Generate a numpy float64 oracle for neuralimagemetrics.pas (FID + IS).

scipy is not available in the project venv, so the FID matrix square-root and
the cross trace Tr(sqrt(Cr*Cg)) are computed directly via numpy eigendecom
position (the same algorithm the Pascal Jacobi solver implements), which is
the documented reference. Synthetic Gaussian feature sets are used for FID;
controlled probability sets for the Inception Score.

Run:  /home/bpsa/x/bin/python tools/make_image_metrics_fixture.py
Writes: tests/fixtures/image_metrics_oracle.json
"""
import json
import os
import numpy as np

np.random.seed(20260614)

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                   "image_metrics_oracle.json")


def matrix_sqrt_spd(A):
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def fid(muR, muG, CR, CG):
    diff = float(np.sum((muR - muG) ** 2))
    sa = matrix_sqrt_spd(CR)
    M = sa @ CG @ sa
    M = 0.5 * (M + M.T)
    w = np.linalg.eigvalsh(M)
    w = np.clip(w, 0, None)
    cross = float(np.sum(np.sqrt(w)))
    val = diff + float(np.trace(CR) + np.trace(CG)) - 2.0 * cross
    return max(val, 0.0)


def sample_cov(X):
    mu = X.mean(axis=0)
    return mu, np.cov(X, rowvar=False, bias=False)


def inception_score(probs, splits):
    n = probs.shape[0]
    scores = []
    for s in range(splits):
        first = (s * n) // splits
        last = ((s + 1) * n) // splits
        part = probs[first:last]
        pbar = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(pbar + 1e-12))
        kl = kl.sum(axis=1)
        scores.append(float(np.exp(kl.mean())))
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())  # population std


cases = []

# Case 1: two distinct Gaussian feature sets, dim 8.
d = 8
A = np.random.randn(d, d) * 0.5
covbaseR = A @ A.T + np.eye(d)
muR = np.random.randn(d) * 2.0
XR = np.random.multivariate_normal(muR, covbaseR, size=500)
B = np.random.randn(d, d) * 0.4
covbaseG = B @ B.T + np.eye(d) * 1.5
muG = muR + np.random.randn(d) * 1.5
XG = np.random.multivariate_normal(muG, covbaseG, size=400)
mR, cR = sample_cov(XR)
mG, cG = sample_cov(XG)
cases.append({
    "name": "gaussian_dim8",
    "featuresR": XR.tolist(),
    "featuresG": XG.tolist(),
    "meanR": mR.tolist(), "covR": cR.tolist(),
    "meanG": mG.tolist(), "covG": cG.tolist(),
    "fid": fid(mR, mG, cR, cG),
    "fid_self": fid(mR, mR, cR, cR),
})

# Case 2: smaller dim 4, fewer samples (still > dim).
d = 4
C = np.random.randn(d, d)
covbaseR = C @ C.T + np.eye(d)
muR = np.random.randn(d)
XR = np.random.multivariate_normal(muR, covbaseR, size=60)
D = np.random.randn(d, d)
covbaseG = D @ D.T + np.eye(d)
muG = np.random.randn(d)
XG = np.random.multivariate_normal(muG, covbaseG, size=50)
mR, cR = sample_cov(XR)
mG, cG = sample_cov(XG)
cases.append({
    "name": "gaussian_dim4",
    "featuresR": XR.tolist(),
    "featuresG": XG.tolist(),
    "meanR": mR.tolist(), "covR": cR.tolist(),
    "meanG": mG.tolist(), "covG": cG.tolist(),
    "fid": fid(mR, mG, cR, cG),
    "fid_self": fid(mR, mR, cR, cR),
})

# Inception score cases.
is_cases = []

# Confident + balanced over 5 classes: 5 samples each one-hot to a distinct
# class -> IS == NumClasses exactly.
K = 5
probs = np.full((K, K), 1e-9)
for i in range(K):
    probs[i] = 1e-9
    probs[i, i] = 1.0
probs = probs / probs.sum(axis=1, keepdims=True)
is_cases.append({
    "name": "confident_balanced_5",
    "probs": probs.tolist(),
    "splits": 1,
    "score": inception_score(probs, 1)[0],
})

# Uniform predictions -> IS == 1 (no information).
probs = np.full((20, 4), 0.25)
is_cases.append({
    "name": "uniform_4",
    "probs": probs.tolist(),
    "splits": 1,
    "score": inception_score(probs, 1)[0],
})

# Mixed realistic-ish predictions, 3 splits.
np.random.seed(7)
logits = np.random.randn(30, 6) * 2.0
probs = np.exp(logits)
probs = probs / probs.sum(axis=1, keepdims=True)
sc, sd = inception_score(probs, 3)
is_cases.append({
    "name": "mixed_6_splits3",
    "probs": probs.tolist(),
    "splits": 3,
    "score": sc,
    "std": sd,
})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"fid_cases": cases, "is_cases": is_cases}, f)
print("wrote", os.path.abspath(OUT))
for c in cases:
    print("FID", c["name"], c["fid"], "self", c["fid_self"])
for c in is_cases:
    print("IS", c["name"], c["score"])
