#!/usr/bin/env python3
"""Generate the COMBINED base-UNet-plus-ControlNet single-step parity fixture
for tests/TestNeuralPretrained.pas (TestControlNetCombinedParity).

This is the end-to-end ControlNet generation oracle: it runs a ControlNet to get
the down_block_res_samples + mid_block_res_sample, then runs the base SD UNet
denoise WITH those residuals injected into the decoder skip connections, exactly
as diffusers' StableDiffusionControlNetPipeline does:

    down_block_res_samples = [d + c for d, c in zip(down, controlnet_down)]
    mid_block_res_sample  += controlnet_mid

It REUSES the two existing pico fixtures (config-compatible: both share
block_out_channels=[16,32], in_channels=4, latent grid 8, text seq 5, cross dim
12). The base UNet weights come from tests/fixtures/tiny_sd_unet.safetensors and
the ControlNet weights from tests/fixtures/tiny_controlnet.safetensors (real
networks also carry independent base + controlnet weights). A single SHARED
(latent, text, control, t) input is pinned and the combined noise prediction is
written to tests/fixtures/tiny_controlnet_combined_io.json.

No new safetensors are produced -- this oracle only adds the combined io.json on
top of the two reused fixtures. The numpy float64 math mirrors the CAI importer
forward EXACTLY (the make_pico recipe); diffusers is not installed.
"""
import json
import numpy as np
from safetensors.numpy import load_file

import sd_unet_tiny_fixture as U
import controlnet_tiny_fixture as C


def load_sd(path):
    return {k: v.astype(np.float64) for k, v in load_file(path).items()}


def forward_with_control(latent, t, enc, sd, down_res, mid_res):
    """Base SD UNet denoise with ControlNet residuals added into the skips +
    mid output. Mirrors U.forward exactly, only injecting the residuals."""
    n = len(U.BLOCK_OUT)
    temb = U.timestep_embedding(t, U.BLOCK_OUT[0])
    temb = temb @ sd['time_embedding.linear_1.weight'].T + sd['time_embedding.linear_1.bias']
    temb = U.silu(temb)
    temb = temb @ sd['time_embedding.linear_2.weight'].T + sd['time_embedding.linear_2.bias']
    h = U.conv2d(latent, sd['conv_in.weight'], sd['conv_in.bias'], 1)
    skips = [h]
    out_ch = U.BLOCK_OUT[0]
    for i, dtype in enumerate(U.DOWN_TYPES):
        out_ch = U.BLOCK_OUT[i]
        is_final = (i == n - 1)
        c = out_ch
        in_ch = skips[-1].shape[0]
        for j in range(U.LAYERS_PER_BLOCK):
            rin = in_ch if j == 0 else out_ch
            h = U.resnet_block(h, temb, sd, f'down_blocks.{i}.resnets.{j}.',
                               rin, out_ch)
            if dtype == 'CrossAttnDownBlock2D':
                h = U.transformer2d(h, enc, sd,
                                    f'down_blocks.{i}.attentions.{j}.', c, U.HEADS)
            skips.append(h)
        if not is_final:
            h = U.conv2d(h, sd[f'down_blocks.{i}.downsamplers.0.conv.weight'],
                         sd[f'down_blocks.{i}.downsamplers.0.conv.bias'], 1, stride=2)
            skips.append(h)
    # ---- inject control residuals into the down-path skips ----
    assert len(skips) == len(down_res), (len(skips), len(down_res))
    skips = [s + r for s, r in zip(skips, down_res)]
    # ---- mid ----
    mid_ch = U.BLOCK_OUT[-1]
    h = U.resnet_block(h, temb, sd, 'mid_block.resnets.0.', mid_ch, mid_ch)
    h = U.transformer2d(h, enc, sd, 'mid_block.attentions.0.', mid_ch, U.HEADS)
    h = U.resnet_block(h, temb, sd, 'mid_block.resnets.1.', mid_ch, mid_ch)
    # ---- inject mid residual ----
    h = h + mid_res
    # ---- up ----
    rev = list(reversed(U.BLOCK_OUT))
    output_channel = rev[0]
    for i, utype in enumerate(U.UP_TYPES):
        prev_output_channel = output_channel
        output_channel = rev[i]
        input_channel = rev[min(i + 1, n - 1)]
        for j in range(U.LAYERS_PER_BLOCK + 1):
            res = skips.pop()
            h = np.concatenate([h, res], axis=0)
            resnet_in = prev_output_channel if j == 0 else output_channel
            res_skip = input_channel if j == U.LAYERS_PER_BLOCK else output_channel
            h = U.resnet_block(h, temb, sd, f'up_blocks.{i}.resnets.{j}.',
                               resnet_in + res_skip, output_channel)
            if utype == 'CrossAttnUpBlock2D':
                h = U.transformer2d(h, enc, sd,
                                    f'up_blocks.{i}.attentions.{j}.',
                                    output_channel, U.HEADS)
        if i != n - 1:
            h = U.upsample_nearest(h)
            h = U.conv2d(h, sd[f'up_blocks.{i}.upsamplers.0.conv.weight'],
                         sd[f'up_blocks.{i}.upsamplers.0.conv.bias'], 1)
    # ---- out ----
    h = U.group_norm(h, sd['conv_norm_out.weight'], sd['conv_norm_out.bias'],
                     U.NORM_GROUPS, U.RES_EPS)
    h = U.silu(h)
    h = U.conv2d(h, sd['conv_out.weight'], sd['conv_out.bias'], 1)
    return h


def main():
    base_sd = load_sd('tests/fixtures/tiny_sd_unet.safetensors')
    ctrl_sd = load_sd('tests/fixtures/tiny_controlnet.safetensors')

    # Pinned SHARED inputs (deterministic dyadic values; exact in f32 + JSON).
    latent = np.zeros((U.IN_CH, U.LATENT, U.LATENT), dtype=np.float64)
    for c in range(U.IN_CH):
        for y in range(U.LATENT):
            for x in range(U.LATENT):
                latent[c, y, x] = (((c * 64 + y * 8 + x) * 7) % 13 - 6) / 8.0
    enc = np.zeros((U.TEXT_SEQ, U.CROSS_DIM), dtype=np.float64)
    for s in range(U.TEXT_SEQ):
        for d in range(U.CROSS_DIM):
            enc[s, d] = (((s * 16 + d) * 5) % 11 - 5) / 8.0
    cond = np.zeros((C.COND_CHANNELS, C.COND_GRID, C.COND_GRID), dtype=np.float64)
    for ch in range(C.COND_CHANNELS):
        for y in range(C.COND_GRID):
            for x in range(C.COND_GRID):
                cond[ch, y, x] = (((ch * 256 + y * 16 + x) * 3) % 17 - 8) / 16.0
    t = 23.0

    down_res, mid_res = C.forward(latent, t, enc, cond, ctrl_sd)
    print(f'controlnet -> {len(down_res)} down residuals '
          f'{[r.shape for r in down_res]}, mid {mid_res.shape}')
    noise = forward_with_control(latent, t, enc, base_sd, down_res, mid_res)
    print(f'combined noise {noise.shape} stats: min {noise.min():.4f} '
          f'max {noise.max():.4f} mean {noise.mean():.4f}')

    # sanity: the control residuals must MATTER (vs the plain base denoise).
    base_only = U.forward(latent, t, enc, base_sd)
    delta = float(np.abs(noise - base_only).max())
    print(f'max |combined - base_only| = {delta:.6f} (must be > 1e-3)')
    assert delta > 1e-3, 'control residuals had no effect on the output'

    with open('tests/fixtures/tiny_controlnet_combined_io.json', 'w') as f:
        json.dump({
            'latent': latent.tolist(),                  # [C][H][W]
            'timestep': t,
            'encoder_hidden_states': enc.tolist(),      # [TEXT_SEQ][CROSS_DIM]
            'controlnet_cond': cond.tolist(),           # [C][CG][CG]
            'noise': noise.tolist(),                    # [OUT_CH][H][W]
            'num_down_residuals': len(down_res),
        }, f)
    print('wrote tests/fixtures/tiny_controlnet_combined_io.json')


if __name__ == '__main__':
    main()
