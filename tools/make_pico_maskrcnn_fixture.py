#!/usr/bin/env python3
"""Generate a tiny RANDOM torchvision-MaskRCNN parity fixture for
tests/TestNeuralPretrained.pas (no network access, torchvision NOT needed).

This is the FIRST instance-segmentation import in the repo. torchvision is not
installed in the reusable venv, so -- exactly like tools/resnet18_tiny_fixture.py
and tools/internlm2_tiny_fixture.py -- the model is a pico random-init network
whose state_dict mirrors torchvision maskrcnn_resnet50_fpn's exact key scheme,
and the reference outputs are computed by a SELF-CONTAINED numpy float64 oracle
of the FPN + RoIAlign + box head + mask head.

Scope (matches the Pascal importer v1, "externally supplied proposal boxes"):
  * The RPN / anchor generator is SKIPPED. The backbone feature maps (two FPN
    input levels, standing in for C4/C5) are supplied directly as inputs and a
    single fixed proposal box is fed to RoIAlign. This is the bounded inference
    vertical the tasklist entry calls for.
  * FPN top-down: lateral 1x1 convs + nearest-neighbour 2x upsample + 3x3
    smoothing convs.
  * RoIAlign (torchvision aligned=True, half-pixel offset; sampling_ratio fixed
    to 2 to match TNNetRoIAlign's serialised box convention).
  * Box head: 2x FC(+ReLU) trunk -> parallel cls_score + bbox_pred.
  * Mask head: 4x (3x3 conv + ReLU) -> ConvTranspose2d(2,stride2)+ReLU -> 1x1
    conv to per-class HxW mask logits.

Weights are round-tripped through float32 before the oracle runs, matching the
CAI loader's float32 read-in (so the < 1e-4 parity tolerance is honest).

torchvision key scheme reproduced (prefixes):
  backbone.fpn.inner_blocks.{i}.0.{weight,bias}   lateral 1x1 conv (level i)
  backbone.fpn.layer_blocks.{i}.0.{weight,bias}   3x3 smoothing conv (level i)
  roi_heads.box_head.fc6.{weight,bias}            FC d_in -> rep
  roi_heads.box_head.fc7.{weight,bias}            FC rep -> rep
  roi_heads.box_predictor.cls_score.{weight,bias} FC rep -> num_classes
  roi_heads.box_predictor.bbox_pred.{weight,bias} FC rep -> num_classes*4
  roi_heads.mask_head.0.0.{weight,bias} ...        4x 3x3 conv  (mask_fcn1..4)
  roi_heads.mask_predictor.conv5_mask.{weight,bias} ConvTranspose2d 2,stride2
  roi_heads.mask_predictor.mask_fcn_logits.{weight,bias} 1x1 -> num_classes
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

# ----- pico hyper-parameters (tiny but exercises every path) -----------------
SEED = 20260626
FPN_OUT = 4          # FPN out_channels
NUM_CLASSES = 3      # incl. background (torchvision: 91 for COCO)
BOX_POOL = 7         # box-head RoIAlign output (torchvision 7)
MASK_POOL = 14       # mask-head RoIAlign output (torchvision 14)
BOX_REP = 8          # box-head representation dim (torchvision 1024)
SAMPLING_RATIO = 2   # torchvision sampling_ratio=2 default

# two FPN input levels standing in for backbone C4, C5.
# (channels in, spatial). chosen so C5 upsamples cleanly onto C4.
LEVELS = [
    ("c4", 6, 8, 8),   # name, in_ch, H, W
    ("c5", 6, 4, 4),
]
# proposal box (feature-map coords on the CHOSEN pyramid level) + level index.
BOX_LEVEL = 0          # pool from P4 (finest level we build)
BOX = (1.3, 0.7, 6.4, 5.9)  # x1,y1,x2,y2 in P4 feature coords

EPS = 1e-5
rng = np.random.RandomState(SEED)


def randw(*shape):
    return (rng.standard_normal(shape) * 0.3).astype(np.float32)


def f32(a):
    return np.asarray(a, dtype=np.float32)


# ----- conv2d (NCHW, single image -> we use CHW), torch-style padding ---------
def conv2d(x, w, b, stride=1, pad=0):
    # x: [C,H,W] float64; w: [O,I,kh,kw]; b: [O]
    C, H, W = x.shape
    O, I, kh, kw = w.shape
    assert I == C
    if pad:
        xp = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float64)
        xp[:, pad:pad + H, pad:pad + W] = x
    else:
        xp = x
    Hp, Wp = xp.shape[1], xp.shape[2]
    Ho = (Hp - kh) // stride + 1
    Wo = (Wp - kw) // stride + 1
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for o in range(O):
        for oy in range(Ho):
            for ox in range(Wo):
                acc = 0.0
                iy0 = oy * stride
                ix0 = ox * stride
                acc = np.sum(w[o] * xp[:, iy0:iy0 + kh, ix0:ix0 + kw])
                out[o, oy, ox] = acc + b[o]
    return out


def relu(x):
    return np.maximum(x, 0.0)


def upsample_nearest2x(x):
    # [C,H,W] -> [C,2H,2W] replicate each pixel into a 2x2 block (DeMaxPool sp=0)
    C, H, W = x.shape
    out = np.zeros((C, 2 * H, 2 * W), dtype=np.float64)
    for c in range(C):
        out[c] = np.repeat(np.repeat(x[c], 2, axis=0), 2, axis=1)
    return out


# ----- RoIAlign (torchvision aligned=True) -----------------------------------
def sample_bilinear(feat, px, py, ci):
    # feat: [C,H,W]; zero-padding out of bounds. Matches TNNetRoIAlign.
    C, H, W = feat.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = x0 + 1
    y1 = y0 + 1
    fx = px - x0
    fy = py - y0
    w00 = (1 - fx) * (1 - fy)
    w01 = fx * (1 - fy)
    w10 = (1 - fx) * fy
    w11 = fx * fy
    v = 0.0
    if 0 <= x0 < W and 0 <= y0 < H:
        v += w00 * feat[ci, y0, x0]
    if 0 <= x1 < W and 0 <= y0 < H:
        v += w01 * feat[ci, y0, x1]
    if 0 <= x0 < W and 0 <= y1 < H:
        v += w10 * feat[ci, y1, x0]
    if 0 <= x1 < W and 0 <= y1 < H:
        v += w11 * feat[ci, y1, x1]
    return v


def roi_align(feat, box, pooled_w, pooled_h, sampling_ratio, spatial_scale=1.0):
    # feat: [C,H,W]; box x1,y1,x2,y2. Returns [C,pooled_h,pooled_w].
    C = feat.shape[0]
    x1, y1, x2, y2 = box
    roi_sx = x1 * spatial_scale - 0.5
    roi_sy = y1 * spatial_scale - 0.5
    roi_w = (x2 - x1) * spatial_scale
    roi_h = (y2 - y1) * spatial_scale
    bin_w = roi_w / pooled_w
    bin_h = roi_h / pooled_h
    if sampling_ratio > 0:
        rx = ry = sampling_ratio
    else:
        rx = max(1, int(np.ceil(roi_w / pooled_w)))
        ry = max(1, int(np.ceil(roi_h / pooled_h)))
    cinv = 1.0 / (rx * ry)
    out = np.zeros((C, pooled_h, pooled_w), dtype=np.float64)
    for ci in range(C):
        for py in range(pooled_h):
            for px in range(pooled_w):
                acc = 0.0
                for iy in range(ry):
                    sy = roi_sy + py * bin_h + (iy + 0.5) * bin_h / ry
                    for ix in range(rx):
                        sx = roi_sx + px * bin_w + (ix + 0.5) * bin_w / rx
                        acc += sample_bilinear(feat, sx, sy, ci)
                out[ci, py, px] = acc * cinv
    return out


def deconv2d_t(x, w, b, stride=2):
    # ConvTranspose2d, torchvision weight [I,O,kh,kw]. x:[I,H,W] -> [O,Ho,Wo]
    I, H, W = x.shape
    Iw, O, kh, kw = w.shape
    assert Iw == I
    Ho = (H - 1) * stride + kh
    Wo = (W - 1) * stride + kw
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for o in range(O):
        for iy in range(H):
            for ix in range(W):
                oy0 = iy * stride
                ox0 = ix * stride
                for c in range(I):
                    out[o, oy0:oy0 + kh, ox0:ox0 + kw] += x[c, iy, ix] * w[c, o]
        out[o] += b[o]
    return out


def main():
    tensors = {}

    # ----- FPN weights: lateral (inner) 1x1 + smoothing (layer) 3x3 ----------
    inner_w = []
    inner_b = []
    layer_w = []
    layer_b = []
    for i, (_, inch, _, _) in enumerate(LEVELS):
        iw = randw(FPN_OUT, inch, 1, 1)
        ib = randw(FPN_OUT)
        lw = randw(FPN_OUT, FPN_OUT, 3, 3)
        lb = randw(FPN_OUT)
        inner_w.append(iw)
        inner_b.append(ib)
        layer_w.append(lw)
        layer_b.append(lb)
        tensors[f"backbone.fpn.inner_blocks.{i}.0.weight"] = iw
        tensors[f"backbone.fpn.inner_blocks.{i}.0.bias"] = ib
        tensors[f"backbone.fpn.layer_blocks.{i}.0.weight"] = lw
        tensors[f"backbone.fpn.layer_blocks.{i}.0.bias"] = lb

    # ----- box head -----------------------------------------------------------
    box_in = BOX_POOL * BOX_POOL * FPN_OUT
    fc6_w = randw(BOX_REP, box_in)
    fc6_b = randw(BOX_REP)
    fc7_w = randw(BOX_REP, BOX_REP)
    fc7_b = randw(BOX_REP)
    cls_w = randw(NUM_CLASSES, BOX_REP)
    cls_b = randw(NUM_CLASSES)
    bbox_w = randw(NUM_CLASSES * 4, BOX_REP)
    bbox_b = randw(NUM_CLASSES * 4)
    tensors["roi_heads.box_head.fc6.weight"] = fc6_w
    tensors["roi_heads.box_head.fc6.bias"] = fc6_b
    tensors["roi_heads.box_head.fc7.weight"] = fc7_w
    tensors["roi_heads.box_head.fc7.bias"] = fc7_b
    tensors["roi_heads.box_predictor.cls_score.weight"] = cls_w
    tensors["roi_heads.box_predictor.cls_score.bias"] = cls_b
    tensors["roi_heads.box_predictor.bbox_pred.weight"] = bbox_w
    tensors["roi_heads.box_predictor.bbox_pred.bias"] = bbox_b

    # ----- mask head: 4x 3x3 conv -> deconv -> 1x1 logits ---------------------
    mask_cw = []
    mask_cb = []
    for k in range(4):
        mw = randw(FPN_OUT, FPN_OUT, 3, 3)
        mb = randw(FPN_OUT)
        mask_cw.append(mw)
        mask_cb.append(mb)
        tensors[f"roi_heads.mask_head.{k}.0.weight"] = mw
        tensors[f"roi_heads.mask_head.{k}.0.bias"] = mb
    deconv_w = randw(FPN_OUT, FPN_OUT, 2, 2)  # ConvTranspose2d [I,O,kh,kw]
    deconv_b = randw(FPN_OUT)
    logits_w = randw(NUM_CLASSES, FPN_OUT, 1, 1)
    logits_b = randw(NUM_CLASSES)
    tensors["roi_heads.mask_predictor.conv5_mask.weight"] = deconv_w
    tensors["roi_heads.mask_predictor.conv5_mask.bias"] = deconv_b
    tensors["roi_heads.mask_predictor.mask_fcn_logits.weight"] = logits_w
    tensors["roi_heads.mask_predictor.mask_fcn_logits.bias"] = logits_b

    # ----- inputs: backbone feature maps for each level ----------------------
    feats_in = []
    feats_json = []
    for (_, inch, H, W) in LEVELS:
        # deterministic-ish but varied values in [-1,1.5]
        a = (rng.standard_normal((inch, H, W)) * 0.5).astype(np.float32)
        feats_in.append(a.astype(np.float64))
        feats_json.append(a.astype(np.float64).tolist())

    # ===== ORACLE forward (float64; weights cast through float32) =============
    def w64(a):
        return f32(a).astype(np.float64)

    # FPN top-down. P5 = lateral(C5); P4 = smooth(lateral(C4)+upsample(latC5)).
    laterals = []
    for i in range(len(LEVELS)):
        laterals.append(conv2d(feats_in[i], w64(inner_w[i]), w64(inner_b[i]),
                               stride=1, pad=0))
    # coarsest level (last) has no top-down add.
    p_levels = [None] * len(LEVELS)
    p_levels[-1] = conv2d(laterals[-1], w64(layer_w[-1]), w64(layer_b[-1]),
                          stride=1, pad=1)
    # the top-down inner feature for the coarsest is its lateral.
    inner = laterals[-1]
    for i in range(len(LEVELS) - 2, -1, -1):
        up = upsample_nearest2x(inner)
        inner = laterals[i] + up
        p_levels[i] = conv2d(inner, w64(layer_w[i]), w64(layer_b[i]),
                             stride=1, pad=1)

    chosen = p_levels[BOX_LEVEL]

    # ----- box head -----------------------------------------------------------
    box_roi = roi_align(chosen, BOX, BOX_POOL, BOX_POOL, SAMPLING_RATIO)
    # torchvision flattens RoIAlign output as [C,H,W] row-major (C-major) then
    # the box_head fc6 weight is [rep, C*H*W] in that same C,H,W order.
    box_flat = box_roi.reshape(-1)  # C-major: c*(H*W)+y*W+x
    h6 = relu(w64(fc6_w) @ box_flat + w64(fc6_b))
    h7 = relu(w64(fc7_w) @ h6 + w64(fc7_b))
    cls_logits = w64(cls_w) @ h7 + w64(cls_b)
    bbox_deltas = w64(bbox_w) @ h7 + w64(bbox_b)

    # ----- mask head ----------------------------------------------------------
    mask_roi = roi_align(chosen, BOX, MASK_POOL, MASK_POOL, SAMPLING_RATIO)
    m = mask_roi
    for k in range(4):
        m = relu(conv2d(m, w64(mask_cw[k]), w64(mask_cb[k]), stride=1, pad=1))
    m = relu(deconv2d_t(m, w64(deconv_w), w64(deconv_b), stride=2))
    mask_logits = conv2d(m, w64(logits_w), w64(logits_b), stride=1, pad=0)

    # ----- write fixture ------------------------------------------------------
    os.makedirs(FIX, exist_ok=True)
    save_file(tensors, os.path.join(FIX, "tiny_maskrcnn.safetensors"))

    config = {
        "model_type": "maskrcnn",
        "architectures": ["MaskRCNN"],
        "fpn_out_channels": FPN_OUT,
        "num_classes": NUM_CLASSES,
        "box_pool_size": BOX_POOL,
        "mask_pool_size": MASK_POOL,
        "box_representation_size": BOX_REP,
        "sampling_ratio": SAMPLING_RATIO,
        "levels": [{"name": n, "in_channels": c, "height": h, "width": w}
                   for (n, c, h, w) in LEVELS],
        "box_level": BOX_LEVEL,
    }
    with open(os.path.join(FIX, "tiny_maskrcnn_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    ref = {
        "feats": feats_json,           # [level][C][H][W]
        "box": list(BOX),
        "box_level": BOX_LEVEL,
        # one P-level row (P4 channel 0) for an FPN sanity check [Hp][Wp]
        "p4_chan0": p_levels[BOX_LEVEL][0].tolist(),
        "cls_logits": cls_logits.tolist(),      # [num_classes]
        "bbox_deltas": bbox_deltas.tolist(),    # [num_classes*4]
        # mask logits [num_classes][Hm][Wm]
        "mask_logits": mask_logits.tolist(),
        "mask_shape": list(mask_logits.shape),
    }
    with open(os.path.join(FIX, "tiny_maskrcnn_ref.json"), "w") as fh:
        json.dump(ref, fh)

    sz = os.path.getsize(os.path.join(FIX, "tiny_maskrcnn.safetensors"))
    rz = os.path.getsize(os.path.join(FIX, "tiny_maskrcnn_ref.json"))
    print("wrote tiny_maskrcnn.safetensors", sz, "bytes")
    print("wrote tiny_maskrcnn_ref.json", rz, "bytes")
    print("mask_logits shape", mask_logits.shape)
    print("cls_logits", np.array(cls_logits))
    print("P-level shapes", [p.shape for p in p_levels])


if __name__ == "__main__":
    main()
