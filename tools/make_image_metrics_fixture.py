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

# ---------------------------------------------------------------------------
# SSIM / MS-SSIM / PSNR oracle. skimage is NOT in the project venv, so this
# hand-writes the 11x11 Gaussian-windowed SSIM exactly as neuralimagemetrics.pas
# implements it: normalised Gaussian window, biased (1/sum-w) local moments,
# 'valid' window map (mean over fully-inside positions). MS-SSIM uses the
# Wang 2003 weights with 2x2 average-pool downsampling between scales.
# ---------------------------------------------------------------------------
WIN = 11
SIGMA = 1.5
K1, K2 = 0.01, 0.03


def gauss_window():
    half = WIN // 2
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * SIGMA * SIGMA))
    return (g / g.sum()).ravel()  # length 121, row-major


def ssim_plane_mean(A, B, C1, C2):
    H, Wd = A.shape
    w = gauss_window()
    oh, ow = H - WIN + 1, Wd - WIN + 1
    acc = 0.0
    for oy in range(oh):
        for ox in range(ow):
            a = A[oy:oy+WIN, ox:ox+WIN].ravel()
            b = B[oy:oy+WIN, ox:ox+WIN].ravel()
            mx = float(np.dot(w, a)); my = float(np.dot(w, b))
            sxx = float(np.dot(w, a*a)) - mx*mx
            syy = float(np.dot(w, b*b)) - my*my
            sxy = float(np.dot(w, a*b)) - mx*my
            a1 = 2*mx*my + C1; a2 = 2*sxy + C2
            b1 = mx*mx + my*my + C1; b2 = sxx + syy + C2
            acc += (a1*a2)/(b1*b2)
    return acc / (oh*ow)


def ssim_plane_lcs(A, B, C1, C2):
    H, Wd = A.shape
    w = gauss_window()
    oh, ow = H - WIN + 1, Wd - WIN + 1
    lsum = 0.0; cssum = 0.0
    for oy in range(oh):
        for ox in range(ow):
            a = A[oy:oy+WIN, ox:ox+WIN].ravel()
            b = B[oy:oy+WIN, ox:ox+WIN].ravel()
            mx = float(np.dot(w, a)); my = float(np.dot(w, b))
            sxx = float(np.dot(w, a*a)) - mx*mx
            syy = float(np.dot(w, b*b)) - my*my
            sxy = float(np.dot(w, a*b)) - mx*my
            lsum += (2*mx*my + C1)/(mx*mx + my*my + C1)
            cssum += (2*sxy + C2)/(sxx + syy + C2)
    return lsum/(oh*ow), cssum/(oh*ow)


def avgpool2x2(P):
    H, Wd = P.shape
    oh, ow = H//2, Wd//2
    P = P[:2*oh, :2*ow]
    return 0.25*(P[0::2,0::2] + P[0::2,1::2] + P[1::2,0::2] + P[1::2,1::2])


def msssim_plane(A, B, C1, C2):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    prod = 1.0
    for s in range(5):
        l, cs = ssim_plane_lcs(A, B, C1, C2)
        if s < 4:
            prod *= cs ** weights[s]
            A = avgpool2x2(A); B = avgpool2x2(B)
        else:
            prod *= (l*cs) ** weights[s]
    return prod


def ssim_multichannel(A, B, C, fn, C1, C2):
    return float(np.mean([fn(A[..., c], B[..., c], C1, C2) for c in range(C)]))


def psnr(A, B, L):
    mse = float(np.mean((A - B)**2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(L*L/mse)


ssim_cases = []
np.random.seed(424242)
# Single-channel 40x40 random pair, DataRange 1.0.
H, Wd = 40, 40
imgA = np.random.rand(H, Wd)
imgB = imgA + 0.15 * np.random.randn(H, Wd)
imgB = np.clip(imgB, 0, 1)
L = 1.0
C1 = (K1*L)**2; C2 = (K2*L)**2
# MS-SSIM needs each scale >= 11; 40 -> 20 -> 10 fails at scale 3. Use a
# larger plane for MS-SSIM (>= 11*2^4 = 176 to be safe over 5 scales).
Hm, Wm = 180, 180
mA = np.random.rand(Hm, Wm)
mB = mA + 0.1 * np.random.randn(Hm, Wm)
mB = np.clip(mB, 0, 1)
ssim_cases.append({
    "name": "gray_40x40",
    "H": H, "W": Wd, "C": 1, "dataRange": L,
    "imgA": imgA.ravel().tolist(),
    "imgB": imgB.ravel().tolist(),
    "ssim": ssim_plane_mean(imgA, imgB, C1, C2),
    "psnr": psnr(imgA, imgB, L),
    "Hm": Hm, "Wm": Wm,
    "imgMA": mA.ravel().tolist(),
    "imgMB": mB.ravel().tolist(),
    "msssim": msssim_plane(mA, mB, C1, C2),
})
# 3-channel 30x30 pair, DataRange 1.0 (SSIM + PSNR only).
H3, W3, Cc = 30, 30, 3
cA = np.random.rand(H3, W3, Cc)
cB = cA + 0.2 * np.random.randn(H3, W3, Cc)
cB = np.clip(cB, 0, 1)
ssim_cases.append({
    "name": "rgb_30x30",
    "H": H3, "W": W3, "C": Cc, "dataRange": L,
    "imgA": cA.ravel().tolist(),   # channel-last flatten
    "imgB": cB.ravel().tolist(),
    "ssim": ssim_multichannel(cA, cB, Cc, ssim_plane_mean, C1, C2),
    "psnr": psnr(cA, cB, L),
})

# ---------------------------------------------------------------------------
# KID (Kernel Inception Distance): unbiased polynomial-kernel MMD^2.
# k(x,y) = (x.y/d + 1)^3; unbiased U-statistic (no diagonal self-terms).
# Validated against the full-set MMD^2 (ComputeKIDMMD2), so no random subset
# bootstrap is needed for parity.
# ---------------------------------------------------------------------------
def poly_kernel_matrix(X, Y):
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** 3


def kid_mmd2(X, Y):
    m, n = X.shape[0], Y.shape[0]
    Kxx = poly_kernel_matrix(X, X)
    Kyy = poly_kernel_matrix(Y, Y)
    Kxy = poly_kernel_matrix(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    termR = Kxx.sum() / (m * (m - 1))
    termG = Kyy.sum() / (n * (n - 1))
    cross = Kxy.sum() / (m * n)
    return float(termR + termG - 2.0 * cross)


kid_cases = []
np.random.seed(20260615)
dk = 16
# Reference set.
XR = np.random.randn(80, dk)
# Same distribution (KID-self / same-dist ~ small).
XGsame = np.random.randn(70, dk)
# Shifted distribution (larger KID).
XGfar = np.random.randn(70, dk) + 3.0
kid_cases.append({
    "name": "gauss_dim16",
    "featuresR": XR.tolist(),
    "featuresGsame": XGsame.tolist(),
    "featuresGfar": XGfar.tolist(),
    "mmd2_self": kid_mmd2(XR, XR),
    "mmd2_same": kid_mmd2(XR, XGsame),
    "mmd2_far": kid_mmd2(XR, XGfar),
})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump({"fid_cases": cases, "is_cases": is_cases,
               "ssim_cases": ssim_cases, "kid_cases": kid_cases}, f)
print("wrote", os.path.abspath(OUT))
for c in ssim_cases:
    print("SSIM", c["name"], c["ssim"], "PSNR", c["psnr"],
          "MSSSIM", c.get("msssim"))
for c in cases:
    print("FID", c["name"], c["fid"], "self", c["fid_self"])
for c in is_cases:
    print("IS", c["name"], c["score"])
