#!/usr/bin/env python3
"""Generate a tiny RANDOM CLIPSeg parity fixture for the COMPLEX transposed-conv
upsample head (use_complex_transposed_convolution=True) for
tests/TestNeuralPretrained.pas.

Companion to tools/clipseg_tiny_fixture.py (the v1 single-ConvTranspose2d head).
Here the decoder head is HF CLIPSegDecoder's complex variant:

    nn.Sequential(
        nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),  # .0
        nn.ReLU(),                                                    # .1
        nn.ConvTranspose2d(reduce_dim, reduce_dim//2, k, stride=k),   # .2
        nn.ReLU(),                                                    # .3
        nn.ConvTranspose2d(reduce_dim//2, 1, k, stride=k),           # .4
    )

with k = patch_size // 4. The CAI importer realizes the 3x3 conv as a
TNNetConvolutionLinear(reduce_dim, 3, pad 1) + TNNetReLU, and each
non-overlapping ConvTranspose2d as PointwiseConvLinear(OutCh*k*k) +
TNNetDepthToSpace(k) + TNNetReLU. Must reproduce the logits < 1e-4.

patch_size = 8 -> k = 2; grid = image_size/patch = 24/8 = 3, so the mask is
3 * 2 * 2 = 12 px square (the complex head output size is grid*k*k, NOT
image_size). The grid must be >= 3 so the leading 3x3 same-conv kernel is not
clamped by CAI's TNNetConvolution (which caps the kernel at the input width).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/clipseg_complex_tiny_fixture.py
writes tests/fixtures/tiny_clipseg_complex{.safetensors,_config.json,_io.json}.
Needs numpy + torch + safetensors + transformers.
"""
import json
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation

torch.manual_seed(20260626)
np.random.seed(20260626)

IMAGE_SIZE = 24
PATCH_SIZE = 8          # grid 3x3 -> 9 patches (+CLS = 10 tokens); k = 8//4 = 2
VISION_HIDDEN = 8
VISION_LAYERS = 3
VISION_HEADS = 2
VISION_INTER = 16
TEXT_HIDDEN = 8
TEXT_LAYERS = 2
TEXT_HEADS = 2
TEXT_INTER = 16
TEXT_VOCAB = 30
TEXT_MAXPOS = 12
PROJECTION_DIM = 6
REDUCE_DIM = 6           # even -> reduce_dim//2 = 3 mid channels
EXTRACT_LAYERS = [0, 1]  # two taps
DECODER_HEADS = 2
DECODER_INTER = 12
LN_EPS = 1e-5
SEQ_LEN = 5              # text prompt length (<= TEXT_MAXPOS)


def build_model():
    cfg = CLIPSegConfig(
        projection_dim=PROJECTION_DIM,
        reduce_dim=REDUCE_DIM,
        extract_layers=EXTRACT_LAYERS,
        decoder_num_attention_heads=DECODER_HEADS,
        decoder_intermediate_size=DECODER_INTER,
        conditional_layer=0,
        use_complex_transposed_convolution=True,
        logit_scale_init_value=2.6592,
        text_config=dict(
            hidden_size=TEXT_HIDDEN,
            intermediate_size=TEXT_INTER,
            num_hidden_layers=TEXT_LAYERS,
            num_attention_heads=TEXT_HEADS,
            vocab_size=TEXT_VOCAB,
            max_position_embeddings=TEXT_MAXPOS,
            layer_norm_eps=LN_EPS,
            hidden_act="quick_gelu",
            eos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=VISION_HIDDEN,
            intermediate_size=VISION_INTER,
            num_hidden_layers=VISION_LAYERS,
            num_attention_heads=VISION_HEADS,
            num_channels=3,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            layer_norm_eps=LN_EPS,
            hidden_act="quick_gelu",
        ),
    )
    model = CLIPSegForImageSegmentation(cfg)
    model.eval()
    # Re-randomize at a healthy O(1) scale (HF init std 0.02 makes the towers
    # near-degenerate so the parity test would not exercise the math).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.25)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.10)
        for name, p in model.named_parameters():
            if name.endswith('layer_norm.weight') or 'layer_norm1.weight' in name \
               or 'layer_norm2.weight' in name or name.endswith('norm1.weight') \
               or name.endswith('norm2.weight') or 'pre_layrnorm.weight' in name \
               or 'post_layernorm.weight' in name or 'final_layer_norm.weight' in name:
                p.add_(1.0)
    return model, cfg


def main():
    model, cfg = build_model()

    sd_f32 = {k: v.detach().to(torch.float32) for k, v in model.state_dict().items()}
    model.load_state_dict(sd_f32)
    model.double()

    pixel = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
    for c in range(3):
        for y in range(IMAGE_SIZE):
            for x in range(IMAGE_SIZE):
                pixel[0, c, y, x] = (((c * 64 + y * IMAGE_SIZE + x) * 5) % 13 - 6) / 8.0
    ids = [0, 7, 11, 4, TEXT_VOCAB - 1]   # last token = highest id = eot
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel,
                    attention_mask=attn, interpolate_pos_encoding=False)
        logits = out.logits[0].cpu().numpy()   # (H, W)
        cond = model.get_conditional_embeddings(
            batch_size=1, input_ids=input_ids, attention_mask=attn).cpu().numpy()[0]

    out_size = logits.shape[0]
    print(f'image {pixel.shape} prompt {ids} -> logits {logits.shape}')
    print(f'logits stats: min {logits.min():.4f} max {logits.max():.4f} '
          f'mean {logits.mean():.4f}')

    save_file(sd_f32, 'tests/fixtures/tiny_clipseg_complex.safetensors')
    config = {
        'model_type': 'clipseg',
        'projection_dim': PROJECTION_DIM,
        'reduce_dim': REDUCE_DIM,
        'extract_layers': EXTRACT_LAYERS,
        'conditional_layer': 0,
        'decoder_num_attention_heads': DECODER_HEADS,
        'decoder_intermediate_size': DECODER_INTER,
        'use_complex_transposed_convolution': True,
        'text_config': {
            'hidden_size': TEXT_HIDDEN,
            'intermediate_size': TEXT_INTER,
            'num_hidden_layers': TEXT_LAYERS,
            'num_attention_heads': TEXT_HEADS,
            'vocab_size': TEXT_VOCAB,
            'max_position_embeddings': TEXT_MAXPOS,
            'layer_norm_eps': LN_EPS,
            'hidden_act': 'quick_gelu',
            'eos_token_id': 2,
        },
        'vision_config': {
            'hidden_size': VISION_HIDDEN,
            'intermediate_size': VISION_INTER,
            'num_hidden_layers': VISION_LAYERS,
            'num_attention_heads': VISION_HEADS,
            'num_channels': 3,
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'layer_norm_eps': LN_EPS,
            'hidden_act': 'quick_gelu',
        },
    }
    with open('tests/fixtures/tiny_clipseg_complex_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_clipseg_complex_io.json', 'w') as f:
        json.dump({
            'pixel': pixel[0].cpu().numpy().tolist(),  # (C,H,W)
            'input_ids': ids,
            'cond': cond.tolist(),
            'logits': logits.tolist(),                 # (H,W)
            'logit_size': out_size,
        }, f)
    n = len(sd_f32)
    print(f'wrote tiny_clipseg_complex.safetensors ({n} tensors) + config + io')

    # ---- fixture self-checks: each complex-head stage must MATTER. ----
    base = logits.copy()

    def perturb(key):
        sd2 = dict(model.state_dict())
        sd2[key] = torch.zeros_like(sd2[key])
        model.load_state_dict(sd2)
        with torch.no_grad():
            l = model(input_ids=input_ids, pixel_values=pixel,
                      attention_mask=attn,
                      interpolate_pos_encoding=False).logits[0].cpu().numpy()
        model.load_state_dict({k: v.double() for k, v in sd_f32.items()})
        return np.abs(l - base).max()

    for key in ('decoder.transposed_convolution.0.weight',
                'decoder.transposed_convolution.0.bias',
                'decoder.transposed_convolution.2.weight',
                'decoder.transposed_convolution.2.bias',
                'decoder.transposed_convolution.4.weight',
                'decoder.transposed_convolution.4.bias',
                'decoder.film_mul.weight',
                'clip.vision_model.encoder.layers.1.self_attn.q_proj.weight'):
        d = perturb(key)
        assert d > 1e-4, f'{key} had no effect ({d})'
        print(f'{key} effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
