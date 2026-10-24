#!/usr/bin/env python3
"""Robust Video Matting parity fixtures for the RVM importer (BuildRVM in
neural/neuralpretrained.pas, tests TestRVM* in tests/TestNeuralPretrained.pas).

Runs the OFFICIAL PeterL1n/RobustVideoMatting model/ package (GPL-3.0; not
vendored here: pass its checkout with --rvm-repo) in float64 and stores, per
frame, the source frame, the clamped fgr/pha and the four recurrent states.

Pico mode (default) builds a random tiny net with the real wiring:
  mobilenetv3 -> tests/fixtures/tiny_rvm_mobilenetv3{.pth,_io.safetensors}
  resnet50    -> tests/fixtures/tiny_rvm_resnet50{.pth,_io.safetensors}
The resnet50 pico keeps torchvision's Bottleneck/_make_layer code (so the
layer4 stride->dilation rule is the real one) with tiny widths and block
counts [2,1,1,2]: layer4 block 0 runs at dilation 1, block 1 at dilation 2.

--refiner fast_guided_filter builds the pico with MattingNetwork(refiner=
'fast_guided_filter') (no refiner weights) into tiny_rvm_<variant>_fgf.*, with
an 'fgf' sequence (ratio 0.5) and a 'seg' sequence (segmentation_pass=True,
ratio 0.5: <prefix>_seg<t> logits + the states).

Real mode (--real CKPT) loads rvm_<variant>.pth and writes one 128x96,
ratio 0.5, 3-frame sequence to --out for a one-off check of the real weights.

Usage (from the repo root; torchvision may live in a separate --tv-path):
  ~/x/bin/python tools/rvm_tiny_fixture.py --rvm-repo DIR --variant resnet50
  ~/x/bin/python tools/rvm_tiny_fixture.py --rvm-repo DIR --variant resnet50 \\
      --real rvm_resnet50.pth --out /tmp/real_io.safetensors

Coded by Claude (AI).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--rvm-repo', required=True,
                help='RobustVideoMatting checkout (holds model/)')
ap.add_argument('--tv-path', default='', help='extra sys.path for torchvision')
ap.add_argument('--variant', choices=['mobilenetv3', 'resnet50'],
                default='resnet50')
ap.add_argument('--refiner', choices=['deep_guided_filter',
                                      'fast_guided_filter'],
                default='deep_guided_filter')
ap.add_argument('--real', default='', help='real rvm_*.pth checkpoint')
ap.add_argument('--out', default='', help='output base (pico) or file (real)')
args = ap.parse_args()
if args.tv_path:
    sys.path.insert(0, args.tv_path)
sys.path.insert(0, args.rvm_repo)

import torch
from torch import nn
from safetensors.torch import save_file
from torchvision.models.resnet import Bottleneck
from model.model import MattingNetwork
from model.resnet import ResNet50Encoder
from model.mobilenetv3 import MobileNetV3LargeEncoder
from model.lraspp import LRASPP
from model.decoder import RecurrentDecoder, Projection
from model.deep_guided_filter import DeepGuidedFilterRefiner
from model.fast_guided_filter import FastGuidedFilterRefiner


class PicoResNetEncoder(ResNet50Encoder):
    """ResNet50Encoder.forward over a tiny torchvision ResNet body."""

    def __init__(self):
        nn.Module.__init__(self)
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 8
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 8, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(Bottleneck, 4, 2)
        self.layer2 = self._make_layer(Bottleneck, 8, 1, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 8, 1, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 16, 2, stride=2, dilate=True)


def pico_mobilenet(m):
    from torchvision.models.mobilenetv3 import MobileNetV3
    from torchvision.models.mobilenetv3 import InvertedResidualConfig as C

    class PicoEncoder(MobileNetV3LargeEncoder):
        def __init__(self):
            MobileNetV3.__init__(self, inverted_residual_setting=[
                C(8, 3, 8, 8, False, "RE", 1, 1, 1),
                C(8, 3, 16, 8, False, "RE", 2, 1, 1),
                C(8, 3, 16, 8, False, "RE", 1, 1, 1),
                C(8, 5, 16, 16, True, "RE", 2, 1, 1),
                C(16, 5, 24, 16, True, "RE", 1, 1, 1),
                C(16, 5, 24, 16, True, "RE", 1, 1, 1),
                C(16, 3, 24, 16, False, "HS", 2, 1, 1),
                C(16, 3, 24, 16, False, "HS", 1, 1, 1),
                C(16, 3, 24, 16, False, "HS", 1, 1, 1),
                C(16, 3, 24, 16, False, "HS", 1, 1, 1),
                C(16, 3, 32, 20, True, "HS", 1, 1, 1),
                C(20, 3, 32, 20, True, "HS", 1, 1, 1),
                C(20, 5, 32, 24, True, "HS", 2, 2, 1),
                C(24, 5, 32, 24, True, "HS", 1, 2, 1),
                C(24, 5, 32, 24, True, "HS", 1, 2, 1),
            ], last_channel=32)
            del self.avgpool
            del self.classifier

    m.backbone = PicoEncoder()
    m.aspp = LRASPP(144, 16)
    m.decoder = RecurrentDecoder([8, 8, 16, 16], [16, 12, 8, 8])


def pico_resnet(m):
    m.backbone = PicoResNetEncoder()
    m.aspp = LRASPP(64, 16)
    m.decoder = RecurrentDecoder([8, 16, 32, 16], [16, 12, 8, 8])


def build_pico():
    torch.manual_seed(1234)
    m = MattingNetwork(args.variant)
    if args.variant == 'mobilenetv3':
        pico_mobilenet(m)
    else:
        pico_resnet(m)
    m.project_mat = Projection(8, 4)
    m.project_seg = Projection(8, 1)
    is_dgf = args.refiner == 'deep_guided_filter'
    m.refiner = DeepGuidedFilterRefiner(8) if is_dgf else FastGuidedFilterRefiner()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.uniform_(0.6, 1.4)
                mod.bias.normal_(0, 0.1)
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
        if is_dgf:
            m.refiner.box_filter.weight.uniform_(0.05, 0.2)
        for p in m.project_mat.parameters():
            p.mul_(0.3)
        m.project_mat.conv.bias[3] += 0.5
    return m.eval()


def store(io, prefix, t, src, fgr, pha, rec):
    io['%s_src%d' % (prefix, t)] = src[0].float().contiguous()
    io['%s_fgr%d' % (prefix, t)] = fgr[0].float().contiguous()
    io['%s_pha%d' % (prefix, t)] = pha[0].float().contiguous()
    for k in range(4):
        io['%s_r%d_%d' % (prefix, k + 1, t)] = rec[k][0].float().contiguous()


def run_random(m, io, g, prefix, W, H, ratio, nframes):
    base = torch.rand(1, 3, H, W, generator=g, dtype=torch.float64)
    rec = [None] * 4
    for t in range(nframes):
        src = (base + 0.15 * torch.rand(1, 3, H, W, generator=g,
                                        dtype=torch.float64)).clamp(0, 1)
        with torch.no_grad():
            fgr, pha, *rec = m(src, *rec, downsample_ratio=ratio)
        store(io, prefix, t, src, fgr, pha, rec)
        print(prefix, t, 'pha range', pha.min().item(), pha.max().item(),
              [tuple(r.shape) for r in rec])


def run_seg(m, io, g, prefix, W, H, ratio, nframes):
    base = torch.rand(1, 3, H, W, generator=g, dtype=torch.float64)
    rec = [None] * 4
    for t in range(nframes):
        src = (base + 0.15 * torch.rand(1, 3, H, W, generator=g,
                                        dtype=torch.float64)).clamp(0, 1)
        with torch.no_grad():
            seg, *rec = m(src, *rec, downsample_ratio=ratio,
                          segmentation_pass=True)
        io['%s_src%d' % (prefix, t)] = src[0].float().contiguous()
        io['%s_seg%d' % (prefix, t)] = seg[0].float().contiguous()
        for k in range(4):
            io['%s_r%d_%d' % (prefix, k + 1, t)] = rec[k][0].float().contiguous()
        print(prefix, t, 'seg range', seg.min().item(), seg.max().item(),
              tuple(seg.shape))


def run_real(m, io):
    W, H = 128, 96
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing='ij')
    g = torch.Generator().manual_seed(3)
    rec = [None] * 4
    for t in range(3):
        bg = torch.stack([xx / W, yy / H, 0.5 + 0.3 * torch.sin(xx / 9)], 0)
        blob = (((xx - 60 - 4 * t) / 22) ** 2 + ((yy - 50) / 30) ** 2 < 1).double()
        person = torch.stack([0.8 + 0 * xx, 0.6 + 0 * xx,
                              0.5 + 0.1 * torch.cos(yy / 5)], 0)
        src = (bg * (1 - blob) + person * blob + 0.02 * torch.rand(
            3, H, W, generator=g, dtype=torch.float64)).clamp(0, 1)[None]
        with torch.no_grad():
            fgr, pha, *rec = m(src, *rec, downsample_ratio=0.5)
        store(io, 'dgf', t, src, fgr, pha, rec)
        print(t, 'pha mean', pha.mean().item())


io = {}
if args.real:
    m = MattingNetwork(args.variant).eval()
    m.load_state_dict(torch.load(args.real, map_location='cpu',
                                 weights_only=True))
    run_real(m.double(), io)
    save_file(io, args.out or '/tmp/rvm_real_io.safetensors')
else:
    out = args.out or 'tests/fixtures/tiny_rvm_' + args.variant
    if args.refiner == 'fast_guided_filter' and not args.out:
        out += '_fgf'
    m = build_pico()
    torch.save(m.state_dict(), out + '.pth')
    m = m.double()
    g = torch.Generator().manual_seed(7)
    if args.refiner == 'fast_guided_filter':
        W, H = (44, 36) if args.variant == 'mobilenetv3' else (80, 64)
        run_random(m, io, g, 'fgf', W, H, 0.5, 2)
        run_seg(m, io, g, 'seg', W, H, 0.5, 2)
    elif args.variant == 'mobilenetv3':
        run_random(m, io, g, 'dgf', 44, 36, 0.5, 3)
        run_random(m, io, g, 'full', 22, 18, 1.0, 2)
    else:
        run_random(m, io, g, 'dgf', 80, 64, 0.5, 3)
        run_random(m, io, g, 'full', 76, 52, 1.0, 2)
    save_file(io, out + '_io.safetensors')
