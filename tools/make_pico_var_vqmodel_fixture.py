#!/usr/bin/env python3
"""Generate a tiny RANDOM diffusers-VQModel fixture MATCHED to the pico VAR
fixture (tools/make_pico_var_fixture.py) so examples/VARGenerate can decode the
VAR loop's final-scale tokens straight to pixels.

The match constraints (vs the generic tools/vqmodel_tiny_fixture.py, which is
sized for the VQModel PARITY test, grid 8 / codebook 16):
  - latent grid == VAR final patch_num (PATCH_NUMS[-1] = 3), so the VAR final
    3x3 token map is exactly a VQ token-id grid;
  - num_vq_embeddings >= VAR vocab (12) so every VAR codebook id is a legal VQ
    id (here 12, exactly the VAR vocab);
  - block_out_channels = [8, 16] -> one upsample -> image grid 3 * 2 = 6.

The forward path is the SAME numpy float64 oracle as the generic VQ fixture
tool; only the sizes differ. The decode is a wiring SMOKE (random weights), not
a real image. Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_var_vqmodel_fixture.py
writes tests/fixtures/tiny_var_vqmodel{.safetensors,_config.json}.
Needs numpy + safetensors only.
"""
import json
import numpy as np
from safetensors.numpy import save_file

EPS = 1e-6
# Matched-to-VAR pico config: latent grid 3 (VAR final patch_num), 1 up block
# -> image 6. Codebook 12 == VAR vocab.
LATENT_GRID = 3        # MUST equal VAR PATCH_NUMS[-1]
IMAGE = LATENT_GRID * 2  # one upsample (2 block_out_channels)
LATENT_CH = 4
VQ_EMBED_DIM = 4
IN_CH = 3
OUT_CH = 3
BLOCK_OUT = [8, 16]
LAYERS_PER_BLOCK = 1
NORM_GROUPS = 4
NUM_VQ_EMBEDDINGS = 12  # MUST be >= VAR vocab (here == 12)

rng = np.random.default_rng(20260626)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def conv_w(out_ch, in_ch, k, std=0.12):
    return randn(out_ch, in_ch, k, k, std=std)


def gn_params(c):
    return randn(c, std=0.3) + 1.0, randn(c, std=0.25)


def add_resnet(sd, prefix, in_ch, out_ch):
    g, b = gn_params(in_ch)
    sd[prefix + 'norm1.weight'], sd[prefix + 'norm1.bias'] = g, b
    sd[prefix + 'conv1.weight'] = conv_w(out_ch, in_ch, 3)
    sd[prefix + 'conv1.bias'] = randn(out_ch, std=0.1)
    g, b = gn_params(out_ch)
    sd[prefix + 'norm2.weight'], sd[prefix + 'norm2.bias'] = g, b
    sd[prefix + 'conv2.weight'] = conv_w(out_ch, out_ch, 3)
    sd[prefix + 'conv2.bias'] = randn(out_ch, std=0.1)
    if in_ch != out_ch:
        sd[prefix + 'conv_shortcut.weight'] = conv_w(out_ch, in_ch, 1)
        sd[prefix + 'conv_shortcut.bias'] = randn(out_ch, std=0.1)


def add_attn(sd, prefix, c):
    g, b = gn_params(c)
    sd[prefix + 'group_norm.weight'], sd[prefix + 'group_norm.bias'] = g, b
    for name in ('to_q', 'to_k', 'to_v'):
        sd[prefix + name + '.weight'] = randn(c, c, std=0.2)
        sd[prefix + name + '.bias'] = randn(c, std=0.1)
    sd[prefix + 'to_out.0.weight'] = randn(c, c, std=0.2)
    sd[prefix + 'to_out.0.bias'] = randn(c, std=0.1)


def build_state_dict():
    sd = {}
    n = len(BLOCK_OUT)
    top = BLOCK_OUT[-1]
    sd['encoder.conv_in.weight'] = conv_w(BLOCK_OUT[0], IN_CH, 3)
    sd['encoder.conv_in.bias'] = randn(BLOCK_OUT[0], std=0.1)
    in_ch = BLOCK_OUT[0]
    for d in range(n):
        out_ch = BLOCK_OUT[d]
        for m in range(LAYERS_PER_BLOCK):
            b = in_ch if m == 0 else out_ch
            add_resnet(sd, f'encoder.down_blocks.{d}.resnets.{m}.', b, out_ch)
        in_ch = out_ch
        if d < n - 1:
            sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.weight'] = \
                conv_w(out_ch, out_ch, 3)
            sd[f'encoder.down_blocks.{d}.downsamplers.0.conv.bias'] = \
                randn(out_ch, std=0.1)
    add_resnet(sd, 'encoder.mid_block.resnets.0.', top, top)
    add_attn(sd, 'encoder.mid_block.attentions.0.', top)
    add_resnet(sd, 'encoder.mid_block.resnets.1.', top, top)
    g, b = gn_params(top)
    sd['encoder.conv_norm_out.weight'], sd['encoder.conv_norm_out.bias'] = g, b
    sd['encoder.conv_out.weight'] = conv_w(LATENT_CH, top, 3)
    sd['encoder.conv_out.bias'] = randn(LATENT_CH, std=0.1)
    sd['quant_conv.weight'] = conv_w(VQ_EMBED_DIM, LATENT_CH, 1)
    sd['quant_conv.bias'] = randn(VQ_EMBED_DIM, std=0.1)
    sd['post_quant_conv.weight'] = conv_w(LATENT_CH, VQ_EMBED_DIM, 1)
    sd['post_quant_conv.bias'] = randn(LATENT_CH, std=0.1)
    emb = randn(NUM_VQ_EMBEDDINGS, VQ_EMBED_DIM, std=0.6)
    sd['quantize.embedding.weight'] = emb
    sd['decoder.conv_in.weight'] = conv_w(top, LATENT_CH, 3)
    sd['decoder.conv_in.bias'] = randn(top, std=0.1)
    add_resnet(sd, 'decoder.mid_block.resnets.0.', top, top)
    add_attn(sd, 'decoder.mid_block.attentions.0.', top)
    add_resnet(sd, 'decoder.mid_block.resnets.1.', top, top)
    in_ch = top
    nres = LAYERS_PER_BLOCK + 1
    for r in range(n):
        rev = n - 1 - r
        out_ch = BLOCK_OUT[rev]
        for m in range(nres):
            b = in_ch if m == 0 else out_ch
            add_resnet(sd, f'decoder.up_blocks.{r}.resnets.{m}.', b, out_ch)
        in_ch = out_ch
        if r < n - 1:
            sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.weight'] = \
                conv_w(out_ch, out_ch, 3)
            sd[f'decoder.up_blocks.{r}.upsamplers.0.conv.bias'] = \
                randn(out_ch, std=0.1)
    g, b = gn_params(BLOCK_OUT[0])
    sd['decoder.conv_norm_out.weight'], sd['decoder.conv_norm_out.bias'] = g, b
    sd['decoder.conv_out.weight'] = conv_w(OUT_CH, BLOCK_OUT[0], 3)
    sd['decoder.conv_out.bias'] = randn(OUT_CH, std=0.1)
    return sd


def main():
    sd = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    save_file(sd_f32, 'tests/fixtures/tiny_var_vqmodel.safetensors')
    config = {
        '_class_name': 'VQModel',
        'block_out_channels': BLOCK_OUT,
        'layers_per_block': LAYERS_PER_BLOCK,
        'latent_channels': LATENT_CH,
        'vq_embed_dim': VQ_EMBED_DIM,
        'num_vq_embeddings': NUM_VQ_EMBEDDINGS,
        'in_channels': IN_CH,
        'out_channels': OUT_CH,
        'norm_num_groups': NORM_GROUPS,
        'sample_size': IMAGE,
        'image_size': IMAGE,
        'latent_size': LATENT_GRID,
        'norm_eps': EPS,
    }
    with open('tests/fixtures/tiny_var_vqmodel_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    print(f'wrote tiny_var_vqmodel.safetensors ({len(sd_f32)} tensors) + config')
    print(f'  latent grid {LATENT_GRID} -> image {IMAGE}, codebook '
          f'{NUM_VQ_EMBEDDINGS}')


if __name__ == '__main__':
    main()
