#!/usr/bin/env python3
"""ParseNet parity fixture for the facexlib face-parsing importer
(BuildParseNet / ParseNetFaceMask in neural/neuralpretrained.pas, tests
TestParseNet* in tests/TestNeuralPretrained.pas).

Runs facexlib 0.3.0's own facexlib/parsing/parsenet.py (not vendored here:
pass the unpacked wheel's directory holding the facexlib package with
--facexlib-dir) in float64 on a pico ParseNet(in_size=32, out_size=32,
min_feat_size=8, base_ch=8, ch_range=[8, 16], res_depth=2, parsing_ch=19), so
the net has both down blocks with and without a width change, identity body
blocks and up blocks. Every BatchNorm gets random affine weights and running
statistics (eval mode), so the importer's BN folding is exercised.

The paste-mask oracle is FaceRestoreHelper.paste_faces_to_input_image's
use_parse branch: MASK_COLORMAP -> cv2.GaussianBlur((101, 101), 11) twice ->
10-pixel border zeroed -> / 255, on two label maps (112 x 112 and 72 x 60, the
second smaller than the kernel, so BORDER_REFLECT_101 reflects repeatedly).
cv2 is used when importable (--extra-path can add e.g. an unpacked
opencv-python-headless); a float64 numpy reflect-101 Gaussian always runs and
must agree with it.

Writes tests/fixtures/tiny_parsenet.pth (plain state_dict, F32, as facexlib's
parsing_parsenet.pth) and tests/fixtures/tiny_parsenet_io.safetensors:
input [3,32,32] in [-1,1], out_mask [19,32,32], labels_a [112,112] /
labels_b [60,72] (class ids as float32) and mask_a / mask_b (same shapes).

Usage (from the repo root):
  ~/x/bin/python -I tools/parsenet_tiny_fixture.py --facexlib-dir /tmp/fxsrc \\
    --extra-path /tmp/cvlib4

Coded by Claude (AI).
"""
import argparse
import importlib.util
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--facexlib-dir', required=True,
                help='directory holding the facexlib package (unpacked wheel)')
ap.add_argument('--extra-path', action='append', default=[],
                help='directory prepended to sys.path (e.g. an unpacked cv2)')
ap.add_argument('--out-dir', default='tests/fixtures')
args = ap.parse_args()
for path in args.extra_path:
    sys.path.insert(0, path)

import numpy as np
import torch
from safetensors.torch import save_file

# parsenet.py imports only numpy and torch; load it without facexlib/__init__.
arch_path = os.path.join(args.facexlib_dir, 'facexlib', 'parsing', 'parsenet.py')
spec = importlib.util.spec_from_file_location('parsenet', arch_path)
parsenet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parsenet)

torch.manual_seed(1234)
rng = np.random.default_rng(1234)
net = parsenet.ParseNet(in_size=32, out_size=32, min_feat_size=8, base_ch=8,
                        parsing_ch=19, res_depth=2, ch_range=[8, 16])
with torch.no_grad():
    for name, tensor in net.state_dict().items():
        if name.endswith('running_mean'):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
        elif name.endswith('running_var'):
            tensor.copy_(torch.rand_like(tensor) * 0.5 + 0.75)
        elif name.endswith('norm.norm.weight'):
            tensor.copy_(torch.rand_like(tensor) * 0.5 + 0.75)
        elif name.endswith('norm.norm.bias'):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
        elif name.endswith('conv2d.weight'):
            fan_in = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
            tensor.copy_(torch.randn_like(tensor) / np.sqrt(fan_in))
        elif name.endswith('conv2d.bias'):
            tensor.copy_(torch.randn_like(tensor) * 0.05)
net.eval()
state = {k: v.float().contiguous() for k, v in net.state_dict().items()
         if not k.endswith('num_batches_tracked')}
# The oracle runs the F32-rounded weights in float64.
net.load_state_dict({k: v.double() for k, v in state.items()}, strict=False)
net.double()

x = torch.rand(1, 3, 32, 32, dtype=torch.float64) * 2 - 1
x = x.float().double()
with torch.no_grad():
    out_mask = net(x)[0]

MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                 255, 255, 0, 255, 0, 0, 0]


def face_labels(height, width):
    """A face-like label map: hair, skin ellipse, eyes, nose, mouth, neck,
    cloth, plus random patches of every class."""
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height * 0.45, width * 0.5
    labels = np.zeros((height, width), np.int64)
    labels[yy < height * 0.3] = 17
    face = ((yy - cy) / (height * 0.35)) ** 2 + ((xx - cx) / (width * 0.3)) ** 2 < 1
    labels[face] = 1
    labels[(yy > height * 0.8) & (np.abs(xx - cx) < width * 0.15)] = 14
    labels[yy > height * 0.9] = 16
    for label, (fy, fx, ry, rx) in {4: (0.38, 0.38, 0.04, 0.07),
                                    5: (0.38, 0.62, 0.04, 0.07),
                                    10: (0.5, 0.5, 0.08, 0.04),
                                    11: (0.65, 0.5, 0.03, 0.1)}.items():
        region = ((yy - fy * height) / (ry * height)) ** 2 + \
                 ((xx - fx * width) / (rx * width)) ** 2 < 1
        labels[region] = label
    for _ in range(12):
        py, px = rng.integers(0, height - 6), rng.integers(0, width - 6)
        labels[py:py + 6, px:px + 6] = rng.integers(0, 19)
    return labels


def gaussian_reflect101(image, ksize, sigma):
    radius = ksize // 2
    taps = np.exp(-(np.arange(ksize) - radius) ** 2 / (2.0 * sigma * sigma))
    taps /= taps.sum()
    padded = np.pad(image, radius, mode='reflect')
    rows = np.zeros((padded.shape[0], image.shape[1]))
    for tap in range(ksize):
        rows += taps[tap] * padded[:, tap:tap + image.shape[1]]
    out = np.zeros(image.shape)
    for tap in range(ksize):
        out += taps[tap] * rows[tap:tap + image.shape[0], :]
    return out


try:
    import cv2
except ImportError:
    cv2 = None
print('cv2:', cv2.__version__ if cv2 is not None else 'not found, numpy only')


def parse_mask(labels):
    """paste_faces_to_input_image's use_parse branch, up to mask / 255."""
    mask = np.zeros(labels.shape)
    for idx, color in enumerate(MASK_COLORMAP):
        mask[labels == idx] = color
    numpy_mask = gaussian_reflect101(gaussian_reflect101(mask, 101, 11), 101, 11)
    if cv2 is not None:
        cv_mask = cv2.GaussianBlur(mask, (101, 101), 11)
        cv_mask = cv2.GaussianBlur(cv_mask, (101, 101), 11)
        diff = np.abs(cv_mask - numpy_mask).max()
        print('  cv2 vs numpy reflect-101 max|diff| (0..255):', diff)
        assert diff < 1e-6, diff
        mask = cv_mask
    else:
        mask = numpy_mask
    thres = 10
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    return mask / 255.


labels_a = face_labels(112, 112)
labels_b = face_labels(60, 72)
io = {
    'input': x[0].float().contiguous(),
    'out_mask': out_mask[0].float().contiguous(),
    'labels_a': torch.from_numpy(labels_a.astype(np.float32)),
    'labels_b': torch.from_numpy(labels_b.astype(np.float32)),
    'mask_a': torch.from_numpy(parse_mask(labels_a).astype(np.float32)),
    'mask_b': torch.from_numpy(parse_mask(labels_b).astype(np.float32)),
}
os.makedirs(args.out_dir, exist_ok=True)
torch.save(state, os.path.join(args.out_dir, 'tiny_parsenet.pth'))
save_file(io, os.path.join(args.out_dir, 'tiny_parsenet_io.safetensors'))
print('out_mask range', float(out_mask.min()), float(out_mask.max()))
print('wrote', len(state), 'tensors')
