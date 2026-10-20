#!/usr/bin/env python3
"""Generate the pico Tiny AutoEncoder (TAESD, diffusers AutoencoderTiny)
DECODER parity fixtures for tests/TestNeuralPretrained.pas.

Two random-weight decoders, one per real latent width:
  tiny_taesd4_decoder*   latent_channels 4  (taesd / taesdxl layout),
                         scaling_factor 1.0, no shift
  tiny_taesd16_decoder*  latent_channels 16 (taesd3 / taef1 layout),
                         scaling_factor 1.5305, shift_factor 0.0609
Each writes:
  <name>.safetensors   decoder.* weights only (F32, exact diffusers keys)
  <name>_config.json   the diffusers config + "latent_size" (latent grid)
  <name>_io.safetensors  "latent" [H,W,C] (denoiser-space latent) and
                         "image" [H,W,3]; both HWC, i.e. CAI volume order
The oracle follows the diffusers pipeline convention:
  image = vae.decode(latent / scaling_factor + shift_factor).sample
run in float64 on the float32-rounded weights.

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  ~/x/bin/python tools/make_pico_taesd_fixture.py
"""
import json
import os

import torch
from diffusers import AutoencoderTiny
from safetensors.torch import save_file

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'tests', 'fixtures')
LATENT = 4                      # latent grid; 2 upsamples -> 16x16 image
BLOCK_OUT = (8, 8, 8)           # decoder_block_out_channels
NUM_BLOCKS = (2, 1, 1)          # num_decoder_blocks

CASES = [
    # name, latent_channels, scaling_factor, shift_factor, seed
    ('tiny_taesd4_decoder', 4, 1.0, 0.0, 20261020),
    ('tiny_taesd16_decoder', 16, 1.5305, 0.0609, 20261021),
]


def build_case(name, latent_ch, scaling, shift, seed):
    torch.manual_seed(seed)
    vae = AutoencoderTiny(
        in_channels=3, out_channels=3,
        encoder_block_out_channels=BLOCK_OUT,
        decoder_block_out_channels=BLOCK_OUT,
        num_encoder_blocks=(1, 1, 1),
        num_decoder_blocks=NUM_BLOCKS,
        latent_channels=latent_ch,
        upsampling_scaling_factor=2,
        act_fn='relu',
        scaling_factor=scaling,
        shift_factor=shift)
    with torch.no_grad():
        # Larger-than-default biases so every bias load matters.
        for pname, p in vae.decoder.named_parameters():
            if pname.endswith('.bias'):
                p.copy_(torch.randn_like(p) * 0.2)
            else:
                p.mul_(1.5)
        for p in vae.parameters():
            p.copy_(p.float())          # round to F32 first
    sd = {'decoder.' + k: v.detach().float().contiguous()
          for k, v in vae.decoder.state_dict().items()}
    save_file(sd, os.path.join(FIXTURE_DIR, name + '.safetensors'))

    cfg = dict(vae.config)
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()
           if not k.startswith('_') or k == '_class_name'}
    cfg['_class_name'] = 'AutoencoderTiny'
    cfg['latent_size'] = LATENT
    with open(os.path.join(FIXTURE_DIR, name + '_config.json'), 'w') as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write('\n')

    # Latent magnitudes up to ~6 so the tanh(x/3)*3 clamp is non-linear.
    latent = (torch.randn(1, latent_ch, LATENT, LATENT) * 2.5).float()
    vae64 = vae.double().eval()
    with torch.no_grad():
        image = vae64.decode(latent.double() / scaling + shift).sample
    assert image.shape == (1, 3, LATENT * 4, LATENT * 4), image.shape
    io = {'latent': latent[0].permute(1, 2, 0).contiguous(),
          'image': image[0].permute(1, 2, 0).float().contiguous()}
    save_file(io, os.path.join(FIXTURE_DIR, name + '_io.safetensors'))
    print(name, 'image range', float(image.min()), float(image.max()),
          'latent maxabs', float(latent.abs().max()))


def main():
    torch.set_num_threads(1)
    for case in CASES:
        build_case(*case)


if __name__ == '__main__':
    main()
