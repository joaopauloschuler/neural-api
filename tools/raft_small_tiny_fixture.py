#!/usr/bin/env python3
"""Generate a tiny RANDOM RAFT (raft_small-style) optical-flow parity fixture
for tests/TestNeuralPretrained.pas.

torchvision is NOT installed in the reusable venv (/home/bpsa/x), so this
script does NOT import torchvision.models.optical_flow.raft_small. Instead it
implements the published RAFT-small forward math directly in plain torch
(Teed & Deng 2020, "RAFT: Recurrent All-Pairs Field Transforms for Optical
Flow", arXiv:2003.12039), at a pico, KB-scale config, with the SAME tensor
names torchvision's raft_small state_dict uses so the Pascal importer
(BuildRaftFromSafeTensors) is a realistic key-for-key parity check. The
reference flow is the float64 oracle this module computes.

Pico architecture (every RAFT-small structural trait, KB-scale):
  * shared FEATURE encoder (fnet) over BOTH frames + a CONTEXT encoder (cnet)
    over frame 1. Each is a SmallEncoder: conv1 7x7 stride-2 + InstanceNorm +
    ReLU, then ONE residual block per stride level down to /4 (pico keeps it
    at /4 instead of /8 to stay tiny), final 1x1 conv. fnet -> feature_dim
    channels; cnet -> hidden_dim + context_dim channels (split into the GRU
    init state tanh(net) and the inp = relu(context)).
  * ALL-PAIRS correlation: corr(i,j) = <f1(i), f2(j)> / sqrt(C). The GRU is
    fed the LOCAL LOOKUP of this volume: for each pixel i, the (2r+1)^2 window
    of correlations corr(i, i + flow(i) + delta), bilinearly sampled. Pico
    uses corr_radius and a SINGLE pyramid level (num_levels = 1).
  * iterative ConvGRU UPDATE over N steps: motion encoder (corr + flow ->
    features), a plain ConvGRU cell over the (H,W,hidden_dim) hidden state,
    and a flow head (2 convs) producing the residual flow delta. flow starts
    at 0 and accumulates the deltas. NO upsampling / convex-mask head (v1 is
    inference-only at /4 resolution; the oracle flow is the coarse field).

The fixture pins: the two input frames (3-channel, small), the predicted
coarse flow field after N iterations.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/raft_small_tiny_fixture.py
writes tests/fixtures/tiny_raft{.safetensors,_config.json,_flow.json}.
Needs torch + numpy + safetensors (NO torchvision).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

# ----------------------- pico config -----------------------
# Frame 32x32 -> /4 = 8x8 flow grid (>= the 7x7 motion-encoder kernel; the CAI
# convolutions CLAMP a kernel to the input spatial size, so the flow grid must
# stay >= 7 for the convf1 7x7 to be a true 7x7 - the ConvNeXt-fixture lesson).
H_IN, W_IN = 32, 32        # input frame size (full res)
STRIDE = 4                  # encoder downsample (pico: /4, real raft is /8)
FEATURE_DIM = 16            # fnet output channels
HIDDEN_DIM = 12             # GRU hidden state channels (net)
CONTEXT_DIM = 8             # context inp channels
CORR_RADIUS = 2             # local lookup radius -> (2r+1)^2 = 25 corr taps
NUM_ITERS = 4               # fixed small iteration count
EPS = 1e-5                  # InstanceNorm epsilon (torch default)


# ------------------- encoder building blocks -------------------
class SmallEncoder(torch.nn.Module):
    """conv1 7x7 s2 + IN + ReLU -> ResBlock(s2) -> ResBlock(s1) -> conv2 1x1.
    Stride from conv1 (2) * first resblock (2) = 4. InstanceNorm (affine)."""

    def __init__(self, out_dim):
        super().__init__()
        c1 = 8
        c2 = 12
        self.conv1 = torch.nn.Conv2d(3, c1, 7, stride=2, padding=3)
        self.norm1 = torch.nn.InstanceNorm2d(c1, affine=True, eps=EPS)
        # residual block 1, stride 2 (downsample), with 1x1 projection shortcut
        self.layer1_conv1 = torch.nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.layer1_norm1 = torch.nn.InstanceNorm2d(c2, affine=True, eps=EPS)
        self.layer1_conv2 = torch.nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.layer1_norm2 = torch.nn.InstanceNorm2d(c2, affine=True, eps=EPS)
        self.layer1_proj = torch.nn.Conv2d(c1, c2, 1, stride=2)
        self.layer1_projnorm = torch.nn.InstanceNorm2d(c2, affine=True, eps=EPS)
        # residual block 2, stride 1 (no downsample, identity shortcut)
        self.layer2_conv1 = torch.nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.layer2_norm1 = torch.nn.InstanceNorm2d(c2, affine=True, eps=EPS)
        self.layer2_conv2 = torch.nn.Conv2d(c2, c2, 3, stride=1, padding=1)
        self.layer2_norm2 = torch.nn.InstanceNorm2d(c2, affine=True, eps=EPS)
        self.conv2 = torch.nn.Conv2d(c2, out_dim, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        # block 1 (downsample)
        y = F.relu(self.layer1_norm1(self.layer1_conv1(x)))
        y = self.layer1_norm2(self.layer1_conv2(y))
        s = self.layer1_projnorm(self.layer1_proj(x))
        x = F.relu(y + s)
        # block 2 (identity)
        y = F.relu(self.layer2_norm1(self.layer2_conv1(x)))
        y = self.layer2_norm2(self.layer2_conv2(y))
        x = F.relu(y + x)
        x = self.conv2(x)
        return x


class MotionEncoder(torch.nn.Module):
    """RAFT-small motion encoder: encode corr -> convc, encode flow -> convf,
    concat -> conv. Output cat with flow feeds the GRU."""

    def __init__(self, corr_dim, mout):
        super().__init__()
        self.convc1 = torch.nn.Conv2d(corr_dim, 16, 1)
        self.convf1 = torch.nn.Conv2d(2, 8, 7, padding=3)
        self.convf2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.conv = torch.nn.Conv2d(16 + 8, mout - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class ConvGRU(torch.nn.Module):
    """Plain ConvGRU cell (RAFT-small SmallUpdateBlock uses a plain ConvGRU,
    not the separable one). hx = [h ; x]; z = sigmoid(Wz*hx); r =
    sigmoid(Wr*hx); q = tanh(Wq*[r*h ; x]); h = (1-z)*h + z*q."""

    def __init__(self, hidden, inp):
        super().__init__()
        self.convz = torch.nn.Conv2d(hidden + inp, hidden, 3, padding=1)
        self.convr = torch.nn.Conv2d(hidden + inp, hidden, 3, padding=1)
        self.convq = torch.nn.Conv2d(hidden + inp, hidden, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class FlowHead(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(hidden, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, h):
        return self.conv2(F.relu(self.conv1(h)))


def build_correlation(f1, f2):
    """All-pairs correlation volume. f1,f2: (1,C,H,W). Returns (H,W,H,W)
    with corr[i,j] = <f1(i),f2(j)> / sqrt(C)."""
    _, c, h, w = f1.shape
    a = f1.view(c, h * w)
    b = f2.view(c, h * w)
    corr = (a.transpose(0, 1) @ b) / (c ** 0.5)   # (HW, HW)
    return corr.view(h, w, h, w)


def lookup(corr, flow, radius):
    """Local lookup of the all-pairs volume around each pixel's current
    target (i + flow). corr: (H,W,H,W). flow: (1,2,H,W) with (dx,dy).
    Returns (1, (2r+1)^2, H, W) bilinearly sampled correlations."""
    h, w = corr.shape[0], corr.shape[1]
    corr2 = corr.view(h * w, 1, h, w)               # batch over source pixel i
    # base target coords = i + flow
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64),
                            indexing="ij")
    cx = xs + flow[0, 0]    # (H,W)
    cy = ys + flow[0, 1]
    taps = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx = cx + dx
            gy = cy + dy
            # normalise to [-1,1] for grid_sample (align_corners=True)
            nx = 2.0 * gx / max(w - 1, 1) - 1.0
            ny = 2.0 * gy / max(h - 1, 1) - 1.0
            grid = torch.stack([nx, ny], dim=-1).view(1, h, w, 2)
            grid = grid.expand(h * w, h, w, 2)
            sampled = F.grid_sample(corr2, grid, mode="bilinear",
                                    align_corners=True, padding_mode="zeros")
            # sampled: (HW,1,H,W); pick the diagonal pixel-i target at (i)
            sampled = sampled.view(h * w, h * w)
            diag = sampled[torch.arange(h * w), torch.arange(h * w)]
            taps.append(diag.view(1, 1, h, w))
    return torch.cat(taps, dim=1)


class PicoRaft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = SmallEncoder(FEATURE_DIM)
        self.cnet = SmallEncoder(HIDDEN_DIM + CONTEXT_DIM)
        corr_dim = (2 * CORR_RADIUS + 1) ** 2
        mout = HIDDEN_DIM           # motion features width fed alongside inp
        self.menc = MotionEncoder(corr_dim, mout)
        self.gru = ConvGRU(HIDDEN_DIM, mout + CONTEXT_DIM)
        self.flow_head = FlowHead(HIDDEN_DIM)

    def forward(self, img1, img2):
        f1 = self.fnet(img1)
        f2 = self.fnet(img2)
        corr = build_correlation(f1, f2)
        ctx = self.cnet(img1)
        net, inp = torch.split(ctx, [HIDDEN_DIM, CONTEXT_DIM], dim=1)
        net = torch.tanh(net)
        inp = F.relu(inp)
        _, _, h, w = f1.shape
        flow = torch.zeros(1, 2, h, w, dtype=torch.float64)
        for _ in range(NUM_ITERS):
            corr_feat = lookup(corr, flow, CORR_RADIUS)
            motion = self.menc(flow, corr_feat)
            x = torch.cat([motion, inp], dim=1)
            net = self.gru(net, x)
            dflow = self.flow_head(net)
            flow = flow + dflow
        return flow


def reinit(model, rng):
    """O(1)-scale re-randomisation so the parity test is not vacuous (the
    ConvNeXt/ModernBERT fixture lesson: std-0.02 init makes norms ~0)."""
    # NOTE (single-precision parity): the CAI runtime is float32 while this
    # oracle is float64. The ConvGRU recurrence is the only SATURATING part of
    # the net (sigmoid/tanh), and a saturated gate amplifies float32 rounding
    # into sign flips that blow past the 1e-4 tolerance over NumIters. So keep
    # every weight SMALL enough that the gates stay in their near-linear regime
    # (|pre-activation| well under 1): a 0.12 std init does this while still
    # exercising the full pipeline (the feature encoder, correlation, lookup,
    # motion encoder and flow head are all LINEAR/ReLU and float32-stable).
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            p.data = torch.tensor(
                rng.standard_normal(tuple(p.shape)) * 0.12, dtype=torch.float64)
        else:
            if "norm" in name and name.endswith(".weight"):
                p.data = torch.tensor(
                    1.0 + rng.standard_normal(tuple(p.shape)) * 0.1,
                    dtype=torch.float64)
            else:
                p.data = torch.tensor(
                    rng.standard_normal(tuple(p.shape)) * 0.1,
                    dtype=torch.float64)


def main():
    rng = np.random.default_rng(20260615)
    model = PicoRaft().double().eval()
    reinit(model, rng)
    # Scale the flow-head output so the predicted flow is O(1 px): big enough
    # to sit well ABOVE the float32 accumulation noise floor (CAI is single
    # precision; a sub-0.01-px flow would be dominated by rounding and fail the
    # 1e-4 parity for the wrong reason) yet small enough to stay on the /4 grid
    # and keep the example warp meaningful.
    with torch.no_grad():
        model.flow_head.conv2.weight.mul_(1.0)
        model.flow_head.conv2.bias.mul_(1.0)

    img1 = np.round(rng.standard_normal((3, H_IN, W_IN)), 4).astype(np.float64)
    img2 = np.round(rng.standard_normal((3, H_IN, W_IN)), 4).astype(np.float64)
    t1 = torch.tensor(img1[None, ...], dtype=torch.float64)
    t2 = torch.tensor(img2[None, ...], dtype=torch.float64)
    with torch.no_grad():
        flow = model(t1, t2).numpy()[0]   # (2, h, w)

    out_sd = {k: v.detach().contiguous().to(torch.float32)
              for k, v in model.state_dict().items()}
    save_file(out_sd, os.path.join(FIX, "tiny_raft.safetensors"))

    config = {
        "model_type": "raft_small",
        "image_height": H_IN, "image_width": W_IN, "stride": STRIDE,
        "feature_dim": FEATURE_DIM, "hidden_dim": HIDDEN_DIM,
        "context_dim": CONTEXT_DIM, "corr_radius": CORR_RADIUS,
        "num_iters": NUM_ITERS, "norm_eps": EPS,
    }
    with open(os.path.join(FIX, "tiny_raft_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    def fmt_chw(arr):
        rows = []
        for c in range(arr.shape[0]):
            rows.append("[" + ",".join(
                "[" + ",".join("%.4f" % v for v in arr[c, y]) + "]"
                for y in range(arr.shape[1])) + "]")
        return "[" + ",".join(rows) + "]"

    with open(os.path.join(FIX, "tiny_raft_flow.json"), "w") as f:
        f.write('{"img1":' + fmt_chw(img1) + ',"img2":' + fmt_chw(img2) +
                ',"flow":' + json.dumps(flow.tolist()) + "}")

    print("wrote tiny_raft.safetensors / _config.json / _flow.json")
    print("flow shape", flow.shape, "flow[:, 0, 0] =", flow[:, 0, 0])
    print("flow dx range", flow[0].min(), flow[0].max())


if __name__ == "__main__":
    main()
