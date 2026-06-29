#!/usr/bin/env python3
"""Generate a tiny RANDOM MaxViT parity fixture for
tests/TestNeuralPretrained.pas, using a SELF-CONTAINED first-principles numpy
float64 oracle (timm is NOT installed offline, and MaxViT is not shipped in
transformers, so -- like the ResNet / MobileNetV3 hand-rolled fixtures -- we
implement the forward pass directly in float64 and pin the reference logits).

The architecture mirrors MaxViT (Tu et al. 2022, *MaxViT: Multi-Axis Vision
Transformer*): a conv stem, then a MaxViT block = MBConv (depthwise-separable
inverted bottleneck + squeeze-excite) -> BLOCK attention (window-local self-
attention over HxW partitioned into local windows) -> GRID attention (the SAME
windowed self-attention applied over a strided/dilated sparse grid of tokens).
Each attention is pre-norm (channel LayerNorm) + relative-position bias + a
post-norm MLP, both with residuals -- exactly the MaxViT MaxViTTransformerBlock.

The ONLY architecturally-new wiring vs the landed Swin importer is the
grid-gather/scatter token permutation: block attention groups HxW into local
windows[wy,wx][iy,ix] = (wy*W+iy, wx*W+ix); grid attention groups it into a
strided grid windows[gy,gx][iy,ix] = (iy*G+gy, ix*G+gx) where G = grid size and
the stride is H/G. Both feed the identical windowed-SDPA + relative-position-bias
path; only the permutation index differs.

Pico config (TINY on purpose -- keeps CAI build under the test memory budget):
  image=8, stem 3x3 stride1 -> 8x8 feature map, stem_ch=4
  MBConv: expand ratio 2 (4->8), depthwise 3x3 stride1 pad1, SE ratio 0.25,
          project 8->6, NO residual (in_ch 4 != out_ch 6)  -> 8x8x6
  block attention: window=4 -> 2x2 windows of 4x4 (16 tokens each)
  grid  attention: grid=4   -> 4x4 windows of 2x2 (stride 2, 4 tokens each)
  heads=2 (head_dim 3), mlp_ratio=2
  head: global-avg-pool -> LayerNorm -> Linear(num_labels=5)

Relative-position bias tables, LayerNorm/SE/classifier biases are all
re-randomized to non-trivial values so every branch measurably moves the logits.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/maxvit_tiny_fixture.py
writes tests/fixtures/tiny_maxvit{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors (no torch / transformers / timm required).
"""
import json
import numpy as np
from safetensors.numpy import save_file

# ---- pico config -----------------------------------------------------------
IMAGE = 8
NUM_CHANNELS = 3
STEM_CH = 4
EXPAND_CH = 8          # MBConv inverted-bottleneck hidden
SE_CH = 2              # squeeze-excite reduced channels
PROJECT_CH = 6         # MBConv output channels == attention dim
DIM = PROJECT_CH
NUM_HEADS = 2
HEAD_DIM = DIM // NUM_HEADS   # 3
WINDOW = 4             # block-attention window size
GRID = 4               # grid-attention grid size (stride = IMAGE // GRID = 2)
MLP_RATIO = 2
MLP_HIDDEN = DIM * MLP_RATIO
NUM_LABELS = 5
LN_EPS = 1e-5
BN_EPS = 1e-5

rng = np.random.default_rng(20260627)


def randn(*shape, std=0.3):
    return (rng.standard_normal(shape) * std).astype(np.float64)


# ---- numpy float64 primitives ---------------------------------------------
def conv2d(x, w, b, stride=1, pad=0):
    # x: (Cin,H,W); w: (Cout,Cin,kh,kw); b: (Cout,)
    Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    if pad:
        xp = np.zeros((Cin, H + 2 * pad, W + 2 * pad), dtype=np.float64)
        xp[:, pad:pad + H, pad:pad + W] = x
        x = xp
        H, W = H + 2 * pad, W + 2 * pad
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    out = np.zeros((Cout, Ho, Wo), dtype=np.float64)
    for o in range(Cout):
        for i in range(Ho):
            for j in range(Wo):
                patch = x[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[o, i, j] = np.sum(patch * w[o]) + b[o]
    return out


def dwconv2d(x, w, b, stride=1, pad=0):
    # depthwise: w (C,1,kh,kw)
    C, H, W = x.shape
    _, _, kh, kw = w.shape
    if pad:
        xp = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float64)
        xp[:, pad:pad + H, pad:pad + W] = x
        x = xp
        H, W = H + 2 * pad, W + 2 * pad
    Ho = (H - kh) // stride + 1
    Wo = (W - kw) // stride + 1
    out = np.zeros((C, Ho, Wo), dtype=np.float64)
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                patch = x[c, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[c, i, j] = np.sum(patch * w[c, 0]) + b[c]
    return out


def gelu(x):
    from math import sqrt
    # exact erf GELU
    from scipy.special import erf  # noqa
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu_noscipy(x):
    # exact erf via numpy (vectorized erf using math.erf)
    import math
    ver = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + ver(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def channel_layernorm(x, gamma, beta, eps=LN_EPS):
    # x: (C,H,W) -> normalize over C at each (h,w)
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * gamma[:, None, None] + beta[:, None, None]


def token_layernorm(x, gamma, beta, eps=LN_EPS):
    # x: (N, C) -> normalize over C per token
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * gamma[None, :] + beta[None, :]


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ---- relative-position index (within a P x P window) -----------------------
def rel_pos_index(P):
    # returns (P*P, P*P) index into a (2P-1)^2 bias table
    coords = np.stack(np.meshgrid(np.arange(P), np.arange(P), indexing='ij'))  # (2,P,P)
    coords = coords.reshape(2, -1)  # (2, P*P)
    rel = coords[:, :, None] - coords[:, None, :]  # (2, P*P, P*P)
    rel = rel + (P - 1)
    idx = rel[0] * (2 * P - 1) + rel[1]
    return idx.astype(np.int64)


# ---- generic windowed multi-head attention with rel-pos bias --------------
def window_attention(tokens, P, qkv_w, qkv_b, proj_w, proj_b, bias_table,
                     rel_idx):
    # tokens: (numwin, P*P, C). qkv_w: (3C, C), qkv_b (3C,). bias_table: ((2P-1)^2, heads)
    numwin, N, C = tokens.shape
    qkv = tokens @ qkv_w.T + qkv_b[None, None, :]      # (numwin,N,3C)
    q, k, v = np.split(qkv, 3, axis=-1)
    # reshape heads
    q = q.reshape(numwin, N, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(numwin, N, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(numwin, N, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale     # (numwin,heads,N,N)
    # relative position bias: table[rel_idx] -> (N,N,heads) -> (heads,N,N)
    bias = bias_table[rel_idx]                          # (N,N,heads)
    bias = bias.transpose(2, 0, 1)[None]               # (1,heads,N,N)
    scores = scores + bias
    attn = softmax(scores, axis=-1)
    out = attn @ v                                     # (numwin,heads,N,N? ) -> (numwin,heads,N,HEAD_DIM)
    out = out.transpose(0, 2, 1, 3).reshape(numwin, N, C)
    out = out @ proj_w.T + proj_b[None, None, :]
    return out


# ---- block / grid partition+merge -----------------------------------------
def block_partition(fmap, P):
    # fmap: (H,W,C) -> (numwin, P*P, C); local PxP windows
    H, W, C = fmap.shape
    nh, nw = H // P, W // P
    out = np.zeros((nh * nw, P * P, C), dtype=np.float64)
    for wy in range(nh):
        for wx in range(nw):
            for iy in range(P):
                for ix in range(P):
                    out[wy * nw + wx, iy * P + ix] = fmap[wy * P + iy, wx * P + ix]
    return out, nh, nw


def block_merge(windows, P, nh, nw, C):
    H, W = nh * P, nw * P
    out = np.zeros((H, W, C), dtype=np.float64)
    for wy in range(nh):
        for wx in range(nw):
            for iy in range(P):
                for ix in range(P):
                    out[wy * P + iy, wx * P + ix] = windows[wy * nw + wx, iy * P + ix]
    return out


def grid_partition(fmap, G):
    # fmap: (H,W,C). grid GxG windows, stride = H//G.
    # window[gy,gx] gathers tokens (iy*G+gy? ) ... MaxViT: grid window indexes
    # positions spaced by G: window[gy,gx][iy,ix] = fmap[iy*sy+gy? ]
    # Use timm convention: reshape (G, sy, G, sx) then attend over the GxG axes.
    # window index (sy_i, sx_i), token within = (gy, gx).
    H, W, C = fmap.shape
    sy, sx = H // G, W // G              # number of windows per side
    # block[a,b] for a in range(sy), b in range(sx): tokens (gy,gx) at (gy*sy+a, gx*sx+b)
    out = np.zeros((sy * sx, G * G, C), dtype=np.float64)
    for a in range(sy):
        for b in range(sx):
            for gy in range(G):
                for gx in range(G):
                    out[a * sx + b, gy * G + gx] = fmap[gy * sy + a, gx * sx + b]
    return out, sy, sx


def grid_merge(windows, G, sy, sx, C):
    H, W = G * sy, G * sx
    out = np.zeros((H, W, C), dtype=np.float64)
    for a in range(sy):
        for b in range(sx):
            for gy in range(G):
                for gx in range(G):
                    out[gy * sy + a, gx * sx + b] = windows[a * sx + b, gy * G + gx]
    return out


# ---- build weights ---------------------------------------------------------
def main():
    W = {}

    # stem conv 3x3 stride1 pad1: 3 -> STEM_CH, with batchnorm folded? Keep plain
    # conv + bias (folded BN at load test == this conv). We emit RAW conv weight
    # + a separate BN that the importer folds. For the oracle apply conv then BN.
    stem_w = randn(STEM_CH, NUM_CHANNELS, 3, 3, std=0.4)
    stem_bn_g = randn(STEM_CH, std=0.3) + 1.0
    stem_bn_b = randn(STEM_CH, std=0.3)
    stem_bn_m = randn(STEM_CH, std=0.2)
    stem_bn_v = np.abs(randn(STEM_CH, std=0.3)) + 0.5
    W['stem.conv.weight'] = stem_w
    W['stem.bn.weight'] = stem_bn_g
    W['stem.bn.bias'] = stem_bn_b
    W['stem.bn.running_mean'] = stem_bn_m
    W['stem.bn.running_var'] = stem_bn_v

    # MBConv expand 1x1 STEM_CH->EXPAND_CH + BN
    exp_w = randn(EXPAND_CH, STEM_CH, 1, 1, std=0.4)
    exp_bn = {k: f(EXPAND_CH) for k, f in [
        ('weight', lambda c: randn(c, std=0.3) + 1.0), ('bias', lambda c: randn(c)),
        ('running_mean', lambda c: randn(c, std=0.2)),
        ('running_var', lambda c: np.abs(randn(c, std=0.3)) + 0.5)]}
    W['mbconv.expand.conv.weight'] = exp_w
    for k, v in exp_bn.items():
        W['mbconv.expand.bn.' + k] = v

    # MBConv depthwise 3x3 stride1 pad1 + BN
    dw_w = randn(EXPAND_CH, 1, 3, 3, std=0.4)
    dw_bn = {k: f(EXPAND_CH) for k, f in [
        ('weight', lambda c: randn(c, std=0.3) + 1.0), ('bias', lambda c: randn(c)),
        ('running_mean', lambda c: randn(c, std=0.2)),
        ('running_var', lambda c: np.abs(randn(c, std=0.3)) + 0.5)]}
    W['mbconv.dw.conv.weight'] = dw_w
    for k, v in dw_bn.items():
        W['mbconv.dw.bn.' + k] = v

    # SE: reduce EXPAND_CH->SE_CH (1x1 conv + bias), expand SE_CH->EXPAND_CH
    se_r_w = randn(SE_CH, EXPAND_CH, 1, 1, std=0.4)
    se_r_b = randn(SE_CH, std=0.3)
    se_e_w = randn(EXPAND_CH, SE_CH, 1, 1, std=0.4)
    se_e_b = randn(EXPAND_CH, std=0.3)
    W['mbconv.se.reduce.weight'] = se_r_w
    W['mbconv.se.reduce.bias'] = se_r_b
    W['mbconv.se.expand.weight'] = se_e_w
    W['mbconv.se.expand.bias'] = se_e_b

    # MBConv project 1x1 EXPAND_CH->PROJECT_CH + BN
    pr_w = randn(PROJECT_CH, EXPAND_CH, 1, 1, std=0.4)
    pr_bn = {k: f(PROJECT_CH) for k, f in [
        ('weight', lambda c: randn(c, std=0.3) + 1.0), ('bias', lambda c: randn(c)),
        ('running_mean', lambda c: randn(c, std=0.2)),
        ('running_var', lambda c: np.abs(randn(c, std=0.3)) + 0.5)]}
    W['mbconv.project.conv.weight'] = pr_w
    for k, v in pr_bn.items():
        W['mbconv.project.bn.' + k] = v

    # block attention: pre-LN, qkv, proj, rel-bias table; post-LN MLP
    def attn_block_weights(prefix, P):
        d = {}
        d[prefix + '.norm1.weight'] = randn(DIM, std=0.3) + 1.0
        d[prefix + '.norm1.bias'] = randn(DIM, std=0.3)
        d[prefix + '.attn.qkv.weight'] = randn(3 * DIM, DIM, std=0.4)
        d[prefix + '.attn.qkv.bias'] = randn(3 * DIM, std=0.3)
        d[prefix + '.attn.proj.weight'] = randn(DIM, DIM, std=0.4)
        d[prefix + '.attn.proj.bias'] = randn(DIM, std=0.3)
        d[prefix + '.attn.rel_pos_bias_table'] = randn((2 * P - 1) ** 2, NUM_HEADS, std=0.6)
        d[prefix + '.norm2.weight'] = randn(DIM, std=0.3) + 1.0
        d[prefix + '.norm2.bias'] = randn(DIM, std=0.3)
        d[prefix + '.mlp.fc1.weight'] = randn(MLP_HIDDEN, DIM, std=0.4)
        d[prefix + '.mlp.fc1.bias'] = randn(MLP_HIDDEN, std=0.3)
        d[prefix + '.mlp.fc2.weight'] = randn(DIM, MLP_HIDDEN, std=0.4)
        d[prefix + '.mlp.fc2.bias'] = randn(DIM, std=0.3)
        return d

    W.update(attn_block_weights('block_attn', WINDOW))
    W.update(attn_block_weights('grid_attn', GRID))

    # head
    W['head.norm.weight'] = randn(DIM, std=0.3) + 1.0
    W['head.norm.bias'] = randn(DIM, std=0.3)
    W['head.fc.weight'] = randn(NUM_LABELS, DIM, std=0.4)
    W['head.fc.bias'] = randn(NUM_LABELS, std=0.2)

    # ---- input pixels (dyadic, exact in f32 + JSON) ------------------------
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for x in range(IMAGE):
                pixels[c, y, x] = (((c * 64 + y * 8 + x) * 5) % 17 - 8) / 8.0

    # ---- ORACLE forward (float64) ------------------------------------------
    def fold_bn(w, g, b, m, v, eps=BN_EPS):
        # returns folded conv weight, bias
        denom = np.sqrt(v + eps)
        scale = g / denom
        nw = w * scale[:, None, None, None]
        nb = b - g * m / denom
        return nw, nb

    # stem
    sw, sb = fold_bn(stem_w, stem_bn_g, stem_bn_b, stem_bn_m, stem_bn_v)
    x = conv2d(pixels, sw, sb, stride=1, pad=1)         # (STEM_CH,8,8)

    # MBConv
    mb_in = x
    ew, eb = fold_bn(exp_w, exp_bn['weight'], exp_bn['bias'],
                     exp_bn['running_mean'], exp_bn['running_var'])
    h = conv2d(mb_in, ew, eb, stride=1, pad=0)
    h = gelu_noscipy(h)
    dwf, dwb = fold_bn(dw_w, dw_bn['weight'], dw_bn['bias'],
                       dw_bn['running_mean'], dw_bn['running_var'])
    h = dwconv2d(h, dwf, dwb, stride=1, pad=1)
    h = gelu_noscipy(h)
    # SE
    pooled = h.mean(axis=(1, 2))                        # (EXPAND_CH,)
    se = conv2d(pooled[:, None, None], se_r_w, se_r_b)[:, 0, 0]
    se = gelu_noscipy(se)
    se = conv2d(se[:, None, None], se_e_w, se_e_b)[:, 0, 0]
    se = sigmoid(se)
    h = h * se[:, None, None]
    # project (no activation)
    pw, pb = fold_bn(pr_w, pr_bn['weight'], pr_bn['bias'],
                     pr_bn['running_mean'], pr_bn['running_var'])
    h = conv2d(h, pw, pb, stride=1, pad=0)             # (PROJECT_CH,8,8)
    # no residual (in 4 != out 6)
    fmap = h.transpose(1, 2, 0)                         # (H,W,C)
    fmap_after_mbconv = fmap.copy()

    def transformer(fmap, prefix, P, partition, merge, rel_idx):
        H, Wd, C = fmap.shape
        tokens_2d = fmap.reshape(H * Wd, C)
        # pre-norm
        n1 = token_layernorm(tokens_2d, W[prefix + '.norm1.weight'],
                             W[prefix + '.norm1.bias']).reshape(H, Wd, C)
        win, p1, p2 = partition(n1, P)
        ao = window_attention(win, P, W[prefix + '.attn.qkv.weight'],
                              W[prefix + '.attn.qkv.bias'],
                              W[prefix + '.attn.proj.weight'],
                              W[prefix + '.attn.proj.bias'],
                              W[prefix + '.attn.rel_pos_bias_table'], rel_idx)
        attn_map = merge(ao, P, p1, p2, C)
        x1 = fmap + attn_map                           # residual
        # MLP post-norm
        t = x1.reshape(H * Wd, C)
        n2 = token_layernorm(t, W[prefix + '.norm2.weight'], W[prefix + '.norm2.bias'])
        m = n2 @ W[prefix + '.mlp.fc1.weight'].T + W[prefix + '.mlp.fc1.bias']
        m = gelu_noscipy(m)
        m = m @ W[prefix + '.mlp.fc2.weight'].T + W[prefix + '.mlp.fc2.bias']
        x2 = t + m
        return x2.reshape(H, Wd, C)

    rel_block = rel_pos_index(WINDOW)
    rel_grid = rel_pos_index(GRID)
    fmap = transformer(fmap, 'block_attn', WINDOW, block_partition, block_merge, rel_block)
    fmap_after_block = fmap.copy()
    fmap = transformer(fmap, 'grid_attn', GRID, grid_partition, grid_merge, rel_grid)
    fmap_after_grid = fmap.copy()

    # head: global avg pool -> token LN -> Linear
    pooled = fmap.reshape(-1, DIM).mean(axis=0)        # (DIM,)
    pn = token_layernorm(pooled[None, :], W['head.norm.weight'], W['head.norm.bias'])[0]
    logits = pn @ W['head.fc.weight'].T + W['head.fc.bias']

    # ---- save safetensors (float32, what CAI loads) ------------------------
    st = {k: v.astype(np.float32) for k, v in W.items()}
    save_file(st, 'tests/fixtures/tiny_maxvit.safetensors')

    cfg = dict(
        image_size=IMAGE, num_channels=NUM_CHANNELS, stem_ch=STEM_CH,
        expand_ch=EXPAND_CH, se_ch=SE_CH, project_ch=PROJECT_CH, dim=DIM,
        num_heads=NUM_HEADS, window=WINDOW, grid=GRID, mlp_ratio=MLP_RATIO,
        num_labels=NUM_LABELS, ln_eps=LN_EPS, bn_eps=BN_EPS)
    with open('tests/fixtures/tiny_maxvit_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    dump = dict(pixels=pixels.tolist(), logits=logits.tolist())
    import os
    if os.environ.get('MAXVIT_DUMP_INTERMEDIATE'):
        dump['mbconv_fmap'] = fmap_after_mbconv.tolist()  # (H,W,C)
        dump['block_fmap'] = fmap_after_block.tolist()
        dump['grid_fmap'] = fmap_after_grid.tolist()
    with open('tests/fixtures/tiny_maxvit_logits.json', 'w') as f:
        json.dump(dump, f)

    print('logits =', logits)
    print('wrote tests/fixtures/tiny_maxvit.{safetensors,_config.json,_logits.json}')


if __name__ == '__main__':
    main()
