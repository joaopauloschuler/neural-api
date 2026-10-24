#!/usr/bin/env python3
"""DBNet (docTR db_mobilenet_v3_large / db_resnet50) parity fixtures for the
DBNet importer (BuildDBNet in neural/neuralpretrained.pas, tests TestDBNet* in
tests/TestNeuralPretrained.pas).

docTR is NOT installed in the venv, so this script re-creates docTR's
DBNet module tree (doctr/models/detection/differentiable_binarization/
pytorch.py: FeaturePyramidNetwork + DBNet.prob_head / thresh_head over an
IntermediateLayerGetter backbone) with the same state_dict tensor names.
The structure and the names are reproduced from docTR's source AS RECALLED;
they have NOT been checked against a released docTR checkpoint yet.

  feat_extractor.*   IntermediateLayerGetter over torchvision
                     mobilenet_v3_large().features (return "3", "6", "12",
                     "16") or resnet50() (return layer1..layer4)
  fpn.in_branches.{i}.0/.1    1x1 conv (no bias) + BatchNorm + ReLU
  fpn.out_branches.{i}.0/.1   3x3 conv (no bias) + BatchNorm + ReLU, then
                              bilinear (align_corners=True) upsample by 2**i
  prob_head.0/.1     3x3 conv (no bias) + BatchNorm + ReLU
  prob_head.3/.4     ConvTranspose2d 2x2 s2 (no bias) + BatchNorm + ReLU
  prob_head.6        ConvTranspose2d 2x2 s2 (bias) -> sigmoid = probability
  thresh_head.*      same layout (training only; the importer ignores it)

The oracle also runs docTR's PreProcessor on a raw 0..255 image: Resize to
input_shape with preserve_aspect_ratio=True, symmetric_pad=True (torchvision
bilinear, antialias=True; zero pad), then Normalize(mean, std).

Pico mode (default) writes, per variant, into tests/fixtures/:
  tiny_dbnet_<variant>.safetensors   float32 state_dict
  tiny_dbnet_<variant>_config.json   docTR-style hub config
  tiny_dbnet_<variant>_io.safetensors  image (3,H0,W0) 0..255 float16, input
      (3,H,W) normalized net input, prob (C,H,W) probability map
torchvision is required (the backbones are the real torchvision modules);
pass its location with --tv-path when it is not on sys.path.

Real mode (--real CKPT --variant V) loads a converted docTR state_dict
(.safetensors or .pt) into the full-size module and writes the same io
tensors for a 1024x1024 synthetic page to --out.

Usage (from the repo root):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \\
    ~/x/bin/python -I tools/dbnet_tiny_fixture.py --tv-path /tmp/tvlib

Coded by Claude (AI).
"""
import argparse
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--tv-path', default='', help='extra sys.path for torchvision')
ap.add_argument('--variant', choices=['mobilenet_v3_large', 'resnet50', 'both'],
                default='both')
ap.add_argument('--real', default='', help='converted docTR checkpoint')
ap.add_argument('--out', default='', help='real mode: io output file')
args = ap.parse_args()
if args.tv_path:
    sys.path.insert(0, args.tv_path)

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from torchvision.models import mobilenet_v3_large, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.mobilenetv3 import MobileNetV3
from torchvision.models.mobilenetv3 import InvertedResidualConfig as IRC
from torchvision.models.resnet import ResNet, Bottleneck

# docTR default_cfgs for both DBNet archs.
DOCTR_MEAN = (0.798, 0.785, 0.772)
DOCTR_STD = (0.264, 0.2749, 0.287)
FPN_LAYERS = {'mobilenet_v3_large': ['3', '6', '12', '16'],
              'resnet50': ['layer1', 'layer2', 'layer3', 'layer4']}


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        out_chans = out_channels // len(in_channels)
        self.in_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chans, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for idx, chans in enumerate(in_channels)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
        self.out_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_chans, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2 ** idx, mode='bilinear',
                            align_corners=True),
            )
            for idx, chans in enumerate(in_channels)
        ])

    def forward(self, x):
        _x = [branch(t) for branch, t in zip(self.in_branches, x)]
        out = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]
        return torch.cat(out, dim=1)


def db_head(head_chans, num_classes):
    return nn.Sequential(
        nn.Conv2d(head_chans, head_chans // 4, 3, padding=1, bias=False),
        nn.BatchNorm2d(head_chans // 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(head_chans // 4, head_chans // 4, 2, stride=2,
                           bias=False),
        nn.BatchNorm2d(head_chans // 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
    )


class DBNet(nn.Module):
    def __init__(self, feat_extractor, head_chans=256, num_classes=1):
        super().__init__()
        self.feat_extractor = feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 224, 224)))
            fpn_channels = [v.shape[1] for _, v in out.items()]
        self.fpn = FeaturePyramidNetwork(fpn_channels, head_chans)
        self.prob_head = db_head(head_chans, num_classes)
        self.thresh_head = db_head(head_chans, num_classes)

    def forward(self, x):
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        return torch.sigmoid(self.prob_head(self.fpn(feats)))


def wrap(backbone, variant):
    return IntermediateLayerGetter(
        backbone, {name: str(idx) for idx, name in enumerate(FPN_LAYERS[variant])})


class PicoResNet(ResNet):
    """torchvision ResNet._make_layer with pico widths (no avgpool / fc)."""

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
        self.layer1 = self._make_layer(Bottleneck, 2, 2)
        self.layer2 = self._make_layer(Bottleneck, 4, 1, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 4, 1, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 6, 1, stride=2)


def pico_mobilenet_features():
    # torchvision mobilenet_v3_large's kernel / SE / activation / stride
    # pattern with pico widths (block 0 keeps exp == in: no expand conv).
    m = MobileNetV3(inverted_residual_setting=[
        IRC(8, 3, 8, 8, False, "RE", 1, 1, 1),
        IRC(8, 3, 16, 8, False, "RE", 2, 1, 1),
        IRC(8, 3, 16, 8, False, "RE", 1, 1, 1),
        IRC(8, 5, 16, 8, True, "RE", 2, 1, 1),
        IRC(8, 5, 16, 8, True, "RE", 1, 1, 1),
        IRC(8, 5, 16, 8, True, "RE", 1, 1, 1),
        IRC(8, 3, 16, 8, False, "HS", 2, 1, 1),
        IRC(8, 3, 16, 8, False, "HS", 1, 1, 1),
        IRC(8, 3, 16, 8, False, "HS", 1, 1, 1),
        IRC(8, 3, 16, 8, False, "HS", 1, 1, 1),
        IRC(8, 3, 16, 16, True, "HS", 1, 1, 1),
        IRC(16, 3, 24, 16, True, "HS", 1, 1, 1),
        IRC(16, 5, 24, 16, True, "HS", 2, 1, 1),
        IRC(16, 5, 24, 16, True, "HS", 1, 1, 1),
        IRC(16, 5, 24, 16, True, "HS", 1, 1, 1),
    ], last_channel=32)
    return m.features


def randomize(m):
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.weight.uniform_(0.6, 1.4)
                mod.bias.normal_(0, 0.2)
                mod.running_mean.normal_(0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)


def doctr_preprocess(image, height, width, mean, std):
    """docTR PreProcessor: aspect-preserving resize, symmetric zero pad, normalize."""
    x = image / 255.0
    h0, w0 = x.shape[-2:]
    target_ratio = height / width
    actual_ratio = h0 / w0
    if actual_ratio > target_ratio:
        tmp_size = (height, max(int(height / actual_ratio), 1))
    else:
        tmp_size = (max(int(width * actual_ratio), 1), width)
    x = F.interpolate(x[None], size=tmp_size, mode='bilinear',
                      align_corners=False, antialias=True)[0]
    pad_w = width - x.shape[-1]
    pad_h = height - x.shape[-2]
    left = math.ceil(pad_w / 2)
    top = math.ceil(pad_h / 2)
    x = F.pad(x, (left, pad_w - left, top, pad_h - top))
    m = torch.tensor(mean, dtype=x.dtype)[:, None, None]
    s = torch.tensor(std, dtype=x.dtype)[:, None, None]
    return (x - m) / s, (left, top, tmp_size[1], tmp_size[0])


def page_image(g, h0, w0):
    """Light page with dark text-line bars plus noise, 0..255."""
    yy, xx = torch.meshgrid(torch.arange(h0, dtype=torch.float64),
                            torch.arange(w0, dtype=torch.float64), indexing='ij')
    img = 220 + 20 * torch.sin(xx / 17) * torch.cos(yy / 13)
    for row in range(3):
        top = 12 + row * (h0 // 4)
        bar = ((yy >= top) & (yy < top + h0 // 10) &
               (xx >= 10 + 7 * row) & (xx < w0 - 15)).double()
        img = img * (1 - bar) + 40 * bar
    img = img[None].repeat(3, 1, 1)
    img = img + 15 * torch.rand(3, h0, w0, generator=g, dtype=torch.float64)
    return img.clamp(0, 255).round()


def run_case(m, io, image, height, width, mean, std):
    net_input, box = doctr_preprocess(image, height, width, mean, std)
    with torch.no_grad():
        prob = m(net_input[None])[0]
    # 0..255 integers are exact in float16, which halves the fixture.
    io['image'] = image.half().contiguous()
    io['input'] = net_input.float().contiguous()
    io['prob'] = prob.float().contiguous()
    io['letterbox'] = torch.tensor(box, dtype=torch.float32)
    print('  letterbox (left, top, w, h) =', box, ' prob range',
          prob.min().item(), prob.max().item(), ' > 0.35:',
          (prob > 0.35).double().mean().item())


def build_pico(variant):
    torch.manual_seed(1234 if variant == 'resnet50' else 4321)
    if variant == 'resnet50':
        backbone = PicoResNet()
    else:
        backbone = pico_mobilenet_features()
    m = DBNet(wrap(backbone, variant), head_chans=16)
    randomize(m)
    return m.eval()


if args.real:
    variant = 'mobilenet_v3_large' if args.variant == 'both' else args.variant
    backbone = (mobilenet_v3_large().features if variant == 'mobilenet_v3_large'
                else resnet50())
    m = DBNet(wrap(backbone, variant)).eval()
    if args.real.endswith('.safetensors'):
        state = load_file(args.real)
    else:
        state = torch.load(args.real, map_location='cpu', weights_only=True)
    m.load_state_dict(state)
    io = {}
    run_case(m.double(), io, page_image(torch.Generator().manual_seed(3),
                                        700, 1000), 1024, 1024,
             DOCTR_MEAN, DOCTR_STD)
    save_file(io, args.out or '/tmp/dbnet_real_io.safetensors')
else:
    variants = (['mobilenet_v3_large', 'resnet50'] if args.variant == 'both'
                else [args.variant])
    # Non-default mean/std/thresholds prove the importer reads the config.
    mean, std = (0.5, 0.45, 0.4), (0.25, 0.3, 0.2)
    height, width = 64, 96
    for variant in variants:
        out = 'tests/fixtures/tiny_dbnet_' + variant
        m = build_pico(variant)
        save_file({k: v.float().contiguous() for k, v in m.state_dict().items()},
                  out + '.safetensors')
        with open(out + '_config.json', 'w') as f:
            json.dump({'arch': 'db_' + variant, 'task': 'detection',
                       'mean': mean, 'std': std,
                       'input_shape': [3, height, width],
                       'classes': ['words'], 'bin_thresh': 0.35,
                       'box_thresh': 0.2}, f, indent=2)
            f.write('\n')
        io = {}
        print(variant)
        run_case(m.double(), io, page_image(torch.Generator().manual_seed(7),
                                            90, 120), height, width, mean, std)
        save_file(io, out + '_io.safetensors')
