#!/usr/bin/env python3
"""Generate a tiny RANDOM CLIPSeg text-prompted segmentation parity fixture for
tests/TestNeuralPretrained.pas.

CLIPSeg (Lueddecke & Ecker 2022, "Image Segmentation Using Text and Image
Prompts", https://arxiv.org/abs/2112.10003): a frozen CLIP ViT image tower whose
intermediate hidden states (config.extract_layers) are projected to reduce_dim,
FiLM-modulated by the CLIP TEXT embedding of a free-text prompt, run through a
small POST-norm transformer decoder, and upsampled by a transposed convolution
to a single-channel HxW logit map (the mask for "whatever the prompt names").

This builds a pico CLIPSeg with RANDOM weights using the REAL HF
transformers.CLIPSegForImageSegmentation as the float64 oracle, then writes the
HF state_dict as a tiny safetensors file plus the expected decoder logit map.
The CAI importer (BuildCLIPSegFromSafeTensors) must reproduce the logits < 1e-4.

We use use_complex_transposed_convolution=False so the upsample is a single
nn.ConvTranspose2d(reduce_dim, 1, patch_size, stride=patch_size) -- exactly a
non-overlapping per-cell projection that the CAI importer realizes as
PointwiseConvLinear(patch*patch) + TNNetDepthToSpace(patch).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/clipseg_tiny_fixture.py
writes tests/fixtures/tiny_clipseg{.safetensors,_config.json,_io.json}.
Needs numpy + torch + safetensors + transformers.
"""
import json
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation

torch.manual_seed(20260615)
np.random.seed(20260615)

IMAGE_SIZE = 8
PATCH_SIZE = 4          # grid 2x2 -> 4 patches (+CLS = 5 tokens)
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
REDUCE_DIM = 6
EXTRACT_LAYERS = [0, 1]   # two taps
DECODER_HEADS = 2
DECODER_INTER = 12
LN_EPS = 1e-5
SEQ_LEN = 5               # text prompt length (<= TEXT_MAXPOS)


def build_model():
    cfg = CLIPSegConfig(
        projection_dim=PROJECTION_DIM,
        reduce_dim=REDUCE_DIM,
        extract_layers=EXTRACT_LAYERS,
        decoder_num_attention_heads=DECODER_HEADS,
        decoder_intermediate_size=DECODER_INTER,
        conditional_layer=0,
        use_complex_transposed_convolution=False,
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
        # LayerNorm weights centered at 1 for a stable forward.
        for name, p in model.named_parameters():
            if name.endswith('layer_norm.weight') or 'layer_norm1.weight' in name \
               or 'layer_norm2.weight' in name or name.endswith('norm1.weight') \
               or name.endswith('norm2.weight') or 'pre_layrnorm.weight' in name \
               or 'post_layernorm.weight' in name or 'final_layer_norm.weight' in name:
                p.add_(1.0)
    return model, cfg


def main():
    model, cfg = build_model()

    # Cast every parameter to float32 (CAI loads float32) then run a float64
    # oracle by upcasting -- matches the importer's f32 storage precision.
    sd_f32 = {k: v.detach().to(torch.float32) for k, v in model.state_dict().items()}
    model.load_state_dict(sd_f32)
    model.double()

    # Pinned inputs.
    pixel = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float64)
    for c in range(3):
        for y in range(IMAGE_SIZE):
            for x in range(IMAGE_SIZE):
                pixel[0, c, y, x] = (((c * 64 + y * 8 + x) * 5) % 13 - 6) / 8.0
    # Prompt token ids; id 2 must be present (CLIP eot = argmax id; force a
    # high id at the last position so the eot pooling is unambiguous).
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
    print(f'cond embed [{PROJECTION_DIM}]: {cond}')

    save_file(sd_f32, 'tests/fixtures/tiny_clipseg.safetensors')
    config = {
        'model_type': 'clipseg',
        'projection_dim': PROJECTION_DIM,
        'reduce_dim': REDUCE_DIM,
        'extract_layers': EXTRACT_LAYERS,
        'conditional_layer': 0,
        'decoder_num_attention_heads': DECODER_HEADS,
        'decoder_intermediate_size': DECODER_INTER,
        'use_complex_transposed_convolution': False,
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
    with open('tests/fixtures/tiny_clipseg_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_clipseg_io.json', 'w') as f:
        json.dump({
            'pixel': pixel[0].cpu().numpy().tolist(),  # (C,H,W)
            'input_ids': ids,
            'cond': cond.tolist(),
            'logits': logits.tolist(),                 # (H,W)
            'logit_size': out_size,
        }, f)
    n = len(sd_f32)
    print(f'wrote tiny_clipseg.safetensors ({n} tensors) + config + io')

    # ---- fixture self-checks: each major piece must MATTER. ----
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

    for key in ('clip.text_projection.weight',
                'decoder.film_mul.weight',
                'decoder.film_add.weight',
                'decoder.reduces.0.weight',
                'decoder.layers.0.mlp.fc1.weight',
                'decoder.transposed_convolution.weight',
                'clip.vision_model.encoder.layers.1.self_attn.q_proj.weight'):
        d = perturb(key)
        assert d > 1e-4, f'{key} had no effect ({d})'
        print(f'{key} effect: max|diff| = {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
