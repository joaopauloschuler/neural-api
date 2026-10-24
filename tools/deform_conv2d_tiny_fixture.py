#!/usr/bin/env python3
"""torchvision.ops.deform_conv2d parity fixture for the external-offset mode of
TNNetDeformableConv (test TestDeformableConvExternalTorchvisionParity in
tests/TestNeuralNumerical.pas). Fixture generation only.

Writes tests/fixtures/deform_conv2d_tiny.safetensors, every tensor float32 in
the CAI volume order:
  input     [H, W, Cin]              7x6 image (W=7, H=6), Cin=4
  weight    [kH, kW, Cin, Cout]      torch weight.permute(2, 3, 1, 0), Cout=6
  bias      [Cout]
  s{S}.offset    [oH, oW, 2*G*kH*kW] torch offset (1, C, oH, oW) permuted
  s{S}.mask      [oH, oW, G*kH*kW]   sigmoid of a normal sample (used as given)
  s{S}.out_mask  [oH, oW, Cout]      deform_conv2d(..., mask=mask)
  s{S}.out_nomask [oH, oW, Cout]     deform_conv2d(...) without mask
for stride S in {1, 2}; 3x3 kernel, padding 1, dilation 1, G=2 offset groups.
Dilation / weight-group cases on the same input, tag T, each with its own
  T.weight  [kH, kW, Cin/WG, Cout]   torch (Cout, Cin/WG, kH, kW) permuted
  T.offset, T.mask, T.out_mask, T.out_nomask   as above
  d2.     dilation 2, stride 1, padding 2, G=2, WG=1
  wg2.    dilation 1, stride 1, padding 1, G=1, WG=2
  d2wg2.  dilation 2, stride 2, padding 1, G=4, WG=2

Usage (from the repo root; torchvision is not in the venv, pass its path):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/deform_conv2d_tiny_fixture.py --tv-path /tmp/tvlib

Coded by Claude (AI).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--tv-path', default='', help='extra sys.path for torchvision')
ap.add_argument('--out', default='tests/fixtures/deform_conv2d_tiny.safetensors')
args = ap.parse_args()
if args.tv_path:
    sys.path.insert(0, args.tv_path)

import torch
import torchvision
from torchvision.ops import deform_conv2d
from safetensors.torch import save_file

torch.manual_seed(20261024)
H, W, CIN, COUT, K, PAD, GROUPS = 6, 7, 4, 6, 3, 1, 2

x = torch.randn(1, CIN, H, W)
weight = torch.randn(COUT, CIN, K, K) * 0.3
bias = torch.randn(COUT) * 0.1


def hwc(t):
    """(1, C, H, W) -> [H, W, C] contiguous."""
    return t[0].permute(1, 2, 0).contiguous()


tensors = {
    'input': hwc(x),
    'weight': weight.permute(2, 3, 1, 0).contiguous(),
    'bias': bias.contiguous(),
}
for stride in (1, 2):
    oh = (H + 2 * PAD - K) // stride + 1
    ow = (W + 2 * PAD - K) // stride + 1
    # Wide enough that some taps leave the image entirely.
    offset = torch.randn(1, 2 * GROUPS * K * K, oh, ow) * 1.5
    mask = torch.sigmoid(torch.randn(1, GROUPS * K * K, oh, ow))
    out_mask = deform_conv2d(x, offset, weight, bias, stride=stride,
                             padding=PAD, mask=mask)
    out_nomask = deform_conv2d(x, offset, weight, bias, stride=stride,
                               padding=PAD)
    tag = 's%d.' % stride
    tensors[tag + 'offset'] = hwc(offset)
    tensors[tag + 'mask'] = hwc(mask)
    tensors[tag + 'out_mask'] = hwc(out_mask)
    tensors[tag + 'out_nomask'] = hwc(out_nomask)

# (tag, dilation, stride, padding, offset groups, weight groups)
for tag, dil, stride, pad, groups, wgroups in (('d2.', 2, 1, 2, 2, 1),
                                               ('wg2.', 1, 1, 1, 1, 2),
                                               ('d2wg2.', 2, 2, 1, 4, 2)):
    ext = dil * (K - 1) + 1
    oh = (H + 2 * pad - ext) // stride + 1
    ow = (W + 2 * pad - ext) // stride + 1
    w = torch.randn(COUT, CIN // wgroups, K, K) * 0.3
    offset = torch.randn(1, 2 * groups * K * K, oh, ow) * 1.5
    mask = torch.sigmoid(torch.randn(1, groups * K * K, oh, ow))
    out_mask = deform_conv2d(x, offset, w, bias, stride=stride, padding=pad,
                             dilation=dil, mask=mask)
    out_nomask = deform_conv2d(x, offset, w, bias, stride=stride, padding=pad,
                               dilation=dil)
    tensors[tag + 'weight'] = w.permute(2, 3, 1, 0).contiguous()
    tensors[tag + 'offset'] = hwc(offset)
    tensors[tag + 'mask'] = hwc(mask)
    tensors[tag + 'out_mask'] = hwc(out_mask)
    tensors[tag + 'out_nomask'] = hwc(out_nomask)

save_file({k: v.float() for k, v in tensors.items()}, args.out)
print('torchvision', torchvision.__version__, '->', args.out)
for k, v in tensors.items():
    print(' ', k, tuple(v.shape))
