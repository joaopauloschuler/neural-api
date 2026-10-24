#!/usr/bin/env python3
"""Generate the pico LaMa (big-lama FFCResNetGenerator) parity fixture for
tests/TestNeuralPretrained.pas.

The lama repo is not pip-installable, so this file re-implements the
generator of saicinpainting/training/modules/ffc.py with the SAME module
names (state_dict keys match anyisalin/big-lama's big-lama.safetensors, with
the "generator." prefix) but tiny widths: ngf 8, 3 downsamplings (bottleneck
64 = 16 local + 48 global, ratio 0.75), 2 FFC resnet blocks, no LFU, sigmoid
output. Input 16x24 gives a 2x3 bottleneck, so the Fourier unit runs on an
odd, non-power-of-two width. Writes:
  tiny_lama.safetensors  generator.* (F32) plus num_batches_tracked (I64) and
                         one val_evaluator.* tensor the builder must skip
  tiny_lama_io.json      config, "input" (H*W*4 floats, HWC = CAI order:
                         image*(1-mask) | mask) and "output" (H*W*3, HWC,
                         the sigmoid prediction before compositing)
The oracle runs in float64 on the float32-rounded weights with random
BatchNorm running statistics (eval mode).

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  ~/x/bin/python tools/make_pico_lama_fixture.py
"""
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'tests', 'fixtures')
NGF = 8
N_DOWN = 3
N_BLOCKS = 2
RATIO = 0.75
HEIGHT, WIDTH = 16, 24
SEED = 20261020


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1,
                                    bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1),
                                norm='ortho')


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    def __init__(self, in_ch, out_ch, k, ratio_gin, ratio_gout, stride,
                 padding):
        super().__init__()
        in_cg = int(in_ch * ratio_gin)
        in_cl = in_ch - in_cg
        out_cg = int(out_ch * ratio_gout)
        out_cl = out_ch - out_cg
        self.ratio_gout = ratio_gout

        def conv(i, o):
            if i == 0 or o == 0:
                return nn.Identity()
            return nn.Conv2d(i, o, k, stride, padding, bias=False,
                             padding_mode='reflect')
        self.convl2l = conv(in_cl, out_cl)
        self.convl2g = conv(in_cl, out_cg)
        self.convg2l = conv(in_cg, out_cl)
        self.convg2g = (nn.Identity() if in_cg == 0 or out_cg == 0
                        else SpectralTransform(in_cg, out_cg))

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_ch, out_ch, k, ratio_gin, ratio_gout, stride=1,
                 padding=0):
        super().__init__()
        self.ffc = FFC(in_ch, out_ch, k, ratio_gin, ratio_gout, stride,
                       padding)
        global_ch = int(out_ch * ratio_gout)
        self.bn_l = (nn.Identity() if ratio_gout == 1
                     else nn.BatchNorm2d(out_ch - global_ch))
        self.bn_g = (nn.Identity() if ratio_gout == 0
                     else nn.BatchNorm2d(global_ch))
        self.act_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(True)
        self.act_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        return self.act_l(self.bn_l(x_l)), self.act_g(self.bn_g(x_g))


class FFCResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, 3, RATIO, RATIO, padding=1)
        self.conv2 = FFC_BN_ACT(dim, dim, 3, RATIO, RATIO, padding=1)

    def forward(self, x):
        x_l, x_g = x
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [nn.ReflectionPad2d(3), FFC_BN_ACT(4, NGF, 7, 0, 0)]
        for i in range(N_DOWN):
            mult = 2 ** i
            gout = RATIO if i == N_DOWN - 1 else 0
            model += [FFC_BN_ACT(NGF * mult, NGF * mult * 2, 3, 0, gout,
                                 stride=2, padding=1)]
        feats = NGF * 2 ** N_DOWN
        model += [FFCResnetBlock(feats) for _ in range(N_BLOCKS)]
        model += [ConcatTupleLayer()]
        for i in range(N_DOWN):
            mult = 2 ** (N_DOWN - i)
            model += [nn.ConvTranspose2d(NGF * mult, NGF * mult // 2, 3,
                                         stride=2, padding=1,
                                         output_padding=1),
                      nn.BatchNorm2d(NGF * mult // 2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(NGF, 3, 7), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def main():
    torch.manual_seed(SEED)
    gen = FFCResNetGenerator()
    with torch.no_grad():
        for m in gen.modules():
            if isinstance(m, nn.BatchNorm2d):
                n = m.num_features
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.3, 0.3)
                m.running_mean.uniform_(-0.3, 0.3)
                m.running_var.uniform_(0.5, 2.0)
                m.num_batches_tracked.fill_(1000)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.bias is not None:
                    m.bias.uniform_(-0.3, 0.3)
        # Widen the pre-sigmoid range so the output is not squashed near 0.5.
        gen.model[-2].weight.mul_(8.0)
    gen.eval()
    sd = {'generator.' + k: v.detach().clone().contiguous()
          for k, v in gen.state_dict().items()}
    sd['val_evaluator.scores.fid.mu'] = torch.zeros(4)
    save_file(sd, os.path.join(FIXTURE_DIR, 'tiny_lama.safetensors'))

    gen = gen.double()
    image = torch.rand(1, 3, HEIGHT, WIDTH, dtype=torch.float64)
    mask = torch.zeros(1, 1, HEIGHT, WIDTH, dtype=torch.float64)
    mask[:, :, 4:11, 6:17] = 1.0
    net_in = torch.cat([image * (1 - mask), mask], dim=1)
    with torch.no_grad():
        out = gen(net_in)
    print('output min/max/mean', out.min().item(), out.max().item(),
          out.mean().item())
    io = {
        'config': {'ngf': NGF, 'n_downsampling': N_DOWN,
                   'n_blocks': N_BLOCKS, 'ratio_g': RATIO,
                   'width': WIDTH, 'height': HEIGHT,
                   'bottleneck_local': (NGF * 2 ** N_DOWN) -
                   int(NGF * 2 ** N_DOWN * RATIO),
                   'bottleneck_global': int(NGF * 2 ** N_DOWN * RATIO)},
        'input': net_in[0].permute(1, 2, 0).reshape(-1).tolist(),
        'output': out[0].permute(1, 2, 0).reshape(-1).tolist(),
    }
    with open(os.path.join(FIXTURE_DIR, 'tiny_lama_io.json'), 'w') as f:
        json.dump(io, f)


if __name__ == '__main__':
    main()
