#!/usr/bin/env python3
"""Generate the tiny TORCH-BIN (.pth) ResNet18 parity fixture used by
tests/TestNeuralPretrained.pas::TestResNetTorchBinPthParity.

Unlike tools/resnet18_tiny_fixture.py (which ships a numpy float64 oracle and
mirrors LEGACY CAI maxpool semantics), this fixture proves the new torchvision
.pth load path: the SAME pico ResNet18 weights are written TWICE -
  tests/fixtures/pico_resnet18.safetensors   (read by TNNetSafeTensorsReader)
  tests/fixtures/pico_resnet18.pth           (torch.save zip, read by the
                                              restricted-unpickler
                                              TNNetTorchBinReader)
The importer reaches the .pth reader transparently because
CreatePretrainedTensorReader now dispatches the .pth extension to
TNNetTorchBinReader (same as it already did for .bin). The test asserts the two
loaded nets produce identical logits, so this fixture needs NO numpy oracle.

The config carries NO "cai_maxpool" key, so the importer uses its DEFAULT
PyTorch (floor-sized) stem maxpool - the setting a real torchvision checkpoint
needs. IMAGE 80 is chosen so that with floor maxpool every spatial map stays
>= 3x3 (stem /2 -> 40, maxpool floor -> 20, layer2/3/4 /2 -> 10,5,3); this keeps
CAI's TNNetConvolution off its kernel-clamp path so the real conv arithmetic is
exercised. blocks_per_stage [1,1,1,1] + tiny widths keep BOTH files small (the
torch.save zip is ~24KB, the safetensors ~11KB).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/resnet18_pico_pth_fixture.py
Needs numpy + safetensors + torch (CPU).
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file

EPS = 1e-5
WIDTHS = [2, 3, 4, 5]            # layer1..layer4 widths (BasicBlock, expansion 1)
BLOCKS = [1, 1, 1, 1]           # one block per stage -> small tensor count
STEM = WIDTHS[0]
NUM_CHANNELS = 3
NUM_CLASSES = 4
IMAGE = 80                      # floor maxpool keeps every map >= 3x3

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_bn(prefix, sd, channels):
    sd[prefix + '.weight'] = randn(channels, std=0.3) + 1.0
    sd[prefix + '.bias'] = randn(channels, std=0.25)
    sd[prefix + '.running_mean'] = randn(channels, std=0.3)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.3)) + 0.5


def make_conv(prefix, sd, out_ch, in_ch, k, std=0.15):
    sd[prefix + '.weight'] = randn(out_ch, in_ch, k, k, std=std)


def make_basic_block(prefix, sd, in_ch, out_ch, stride):
    make_conv(prefix + '.conv1', sd, out_ch, in_ch, 3)
    make_bn(prefix + '.bn1', sd, out_ch)
    make_conv(prefix + '.conv2', sd, out_ch, out_ch, 3)
    make_bn(prefix + '.bn2', sd, out_ch)
    if stride != 1 or in_ch != out_ch:
        make_conv(prefix + '.downsample.0', sd, out_ch, in_ch, 1)
        make_bn(prefix + '.downsample.1', sd, out_ch)


def main():
    sd = {}
    make_conv('conv1', sd, STEM, NUM_CHANNELS, 7)
    make_bn('bn1', sd, STEM)
    in_ch = STEM
    for li, (width, nblocks) in enumerate(zip(WIDTHS, BLOCKS), start=1):
        stride = 1 if li == 1 else 2
        for bi in range(nblocks):
            s = stride if bi == 0 else 1
            make_basic_block(f'layer{li}.{bi}', sd, in_ch, width, s)
            in_ch = width
    sd['fc.weight'] = randn(NUM_CLASSES, WIDTHS[-1], std=0.4)
    sd['fc.bias'] = randn(NUM_CLASSES, std=0.3)

    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    save_file(sd_f32, 'tests/fixtures/pico_resnet18.safetensors')

    # Faithful torchvision-style state_dict zip: same float32 storages.
    tsd = {k: torch.from_numpy(np.array(v)) for k, v in sd_f32.items()}
    torch.save(tsd, 'tests/fixtures/pico_resnet18.pth')

    config = {
        'model_type': 'resnet',
        'depth': 18,
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'widths': WIDTHS,
        'blocks_per_stage': BLOCKS,
        'bn_eps': EPS,
        # NO cai_maxpool key -> importer default = PyTorch floor-sized stem
        # maxpool (the real-torchvision setting this fixture exercises).
    }
    with open('tests/fixtures/pico_resnet18_config.json', 'w') as f:
        json.dump(config, f, indent=1)

    print('tensors', len(sd_f32),
          'safetensors', os.path.getsize('tests/fixtures/pico_resnet18.safetensors'),
          'pth', os.path.getsize('tests/fixtures/pico_resnet18.pth'))


if __name__ == '__main__':
    main()
