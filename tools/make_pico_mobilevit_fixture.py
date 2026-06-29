#!/usr/bin/env python3
"""Generate a tiny RANDOM MobileViT parity fixture for
tests/TestNeuralPretrained.pas.

transformers is NOT installed in the reusable venv, so the oracle is a
self-contained numpy float64 first-principles forward of a pico random-init
MobileViT whose state_dict mirrors transformers' exact key scheme
(mobilevit.conv_stem.*, mobilevit.encoder.layer.{i}.*, mobilevit.conv_1x1_exp.*,
classifier.*). The pico net keeps the FIXED mobilevit topology (conv stem; two
MobileNet inverted-residual stages with 1 and 3 residuals; three MobileViT
blocks each with a stride-2 downsample inverted residual + local conv + patch
transformer + fold + fusion) but shrinks every width and uses 1 transformer
layer per MobileViT stage.

This exercises EVERY branch of BuildMobileViT (neuralpretrained.pas):
  - conv_stem 3x3 stride2 + folded BN + SiLU
  - MobileViTInvertedResidual: expand 1x1 (+BN+SiLU), depthwise 3x3 (+BN+SiLU),
    reduce 1x1 (+BN, NO act), residual when stride1 & in==out
  - MobileViTLayer: downsample inv-res -> conv_kxk (+BN+SiLU) -> conv_1x1
    (Hidden, NO BN, NO act) -> UNFOLD (2x2 patches) -> L pre-LN transformer
    layers (block-diagonal per-sub-position attention) -> post LN -> FOLD ->
    conv_projection 1x1 (+BN+SiLU) -> fusion concat(residual, proj) + conv_kxk
  - conv_1x1_exp 1x1 (+BN+SiLU) -> global avg pool -> classifier Linear

The highest-risk part is the unfold/fold. HF MobileViTLayer.unfolding produces
a (B*patch_area, num_patches, C) token tensor where the patch_area=4 intra-cell
sub-positions become INDEPENDENT attention sequences (split onto the batch
axis). The CAI port lays those as one length (patch_area*num_patches) X-axis
sequence ordered s = a*num_patches + p, and makes attention block-diagonal over
the patch_area groups via the SDPA segment side channel. This oracle reproduces
the HF unfold/attention/fold math directly (4 separate (num_patches, C)
sequences) and the parity test compares the CAI net's logits to it.

Square-map note: CAI's TNNetAvgChannel divides by W*W (true global mean only on
square maps). Every spatial map here is square, so the oracle's .mean() matches.

BatchNorm2d eps is HF's hardcoded 1e-5; the importer folds each BN into its
conv at load. LayerNorm eps = layer_norm_eps = 1e-5.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_mobilevit_fixture.py
writes tests/fixtures/tiny_mobilevit{.safetensors,_config.json,_logits.json}.
Needs numpy + safetensors only.
"""
import json
import math
import numpy as np
from safetensors.numpy import save_file

BN_EPS = 1e-5
LN_EPS = 1e-5
# Image 128 keeps every spatial map square and >= 4 at the smallest MobileViT
# block (post-downsample 4x4). CAI's TNNetConvolutionLinear has a small-map
# padding quirk that GROWS a K=3/stride1 conv's output on maps < 4 (e.g. 2->3),
# which would desync the unfold/fold; staying >= 4 avoids it and matches the
# standard PyTorch "same" conv this oracle uses. Map sizes: stem/2 -> 64, l0 ->
# 64, l1/2 -> 32, block0/2 -> 16, block1/2 -> 8, block2/2 -> 4.
IMAGE = 128
NUM_CHANNELS = 3
NUM_CLASSES = 5
PATCH = 2
NUM_HEADS = 2
MLP_RATIO = 2.0
EXPAND_RATIO = 2.0
CONV_K = 3
NECK = [4, 6, 8, 10, 12, 14, 16]   # neck_hidden_sizes
HIDDEN = [8, 10, 12]               # hidden_sizes (transformer widths)
TX_LAYERS = [1, 1, 1]              # transformer layers per MobileViT block

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


# --------------------------------------------------------------------------
# state_dict construction (transformers mobilevit key scheme)
# --------------------------------------------------------------------------
def make_bn(prefix, sd, channels):
    sd[prefix + '.weight'] = randn(channels, std=0.3) + 1.0
    sd[prefix + '.bias'] = randn(channels, std=0.25)
    sd[prefix + '.running_mean'] = randn(channels, std=0.3)
    sd[prefix + '.running_var'] = np.abs(randn(channels, std=0.3)) + 0.5
    sd[prefix + '.num_batches_tracked'] = np.array(0, dtype=np.int64)


def make_convlayer(prefix, sd, out_ch, in_ch, k, depthwise=False, bn=True,
                   std=0.2):
    if depthwise:
        sd[prefix + '.convolution.weight'] = randn(out_ch, 1, k, k, std=0.3)
    else:
        sd[prefix + '.convolution.weight'] = randn(out_ch, in_ch, k, k, std=std)
    if bn:
        make_bn(prefix + '.normalization', sd, out_ch)


def make_invres(prefix, sd, in_ch, out_ch):
    # prefix has NO trailing dot (matches the importer's BnPrefix scheme).
    exp = make_divisible(in_ch * EXPAND_RATIO, 8)
    has_expand = exp != in_ch
    if has_expand:
        make_convlayer(prefix + '.expand_1x1', sd, exp, in_ch, 1)
    make_convlayer(prefix + '.conv_3x3', sd, exp, exp, CONV_K, depthwise=True)
    make_convlayer(prefix + '.reduce_1x1', sd, out_ch, exp, 1)
    return exp


def make_linear(prefix, sd, out_dim, in_dim, std=0.3, bias=True):
    sd[prefix + '.weight'] = randn(out_dim, in_dim, std=std)
    if bias:
        sd[prefix + '.bias'] = randn(out_dim, std=0.2)


def make_ln(prefix, sd, dim):
    sd[prefix + '.weight'] = randn(dim, std=0.2) + 1.0
    sd[prefix + '.bias'] = randn(dim, std=0.2)


def make_txlayer(prefix, sd, hidden):
    inter = int(hidden * MLP_RATIO)
    make_ln(prefix + 'layernorm_before', sd, hidden)
    make_linear(prefix + 'attention.attention.query', sd, hidden, hidden)
    make_linear(prefix + 'attention.attention.key', sd, hidden, hidden)
    make_linear(prefix + 'attention.attention.value', sd, hidden, hidden)
    make_linear(prefix + 'attention.output.dense', sd, hidden, hidden)
    make_ln(prefix + 'layernorm_after', sd, hidden)
    make_linear(prefix + 'intermediate.dense', sd, inter, hidden)
    make_linear(prefix + 'output.dense', sd, hidden, inter)


def make_mvit_block(prefix, sd, in_ch, out_ch, hidden, ntx):
    # prefix HAS a trailing dot (matches the importer).
    make_invres(prefix + 'downsampling_layer', sd, in_ch, out_ch)
    make_convlayer(prefix + 'conv_kxk', sd, out_ch, out_ch, CONV_K)
    make_convlayer(prefix + 'conv_1x1', sd, hidden, out_ch, 1, bn=False)
    for j in range(ntx):
        make_txlayer(prefix + f'transformer.layer.{j}.', sd, hidden)
    make_ln(prefix + 'layernorm', sd, hidden)
    make_convlayer(prefix + 'conv_projection', sd, out_ch, hidden, 1)
    make_convlayer(prefix + 'fusion', sd, out_ch, 2 * out_ch, CONV_K)


def build_state_dict():
    sd = {}
    make_convlayer('mobilevit.conv_stem', sd, NECK[0], NUM_CHANNELS, CONV_K)
    # layer.0: 1 inverted residual NECK0->NECK1 stride1
    make_invres('mobilevit.encoder.layer.0.layer.0', sd, NECK[0], NECK[1])
    # layer.1: 3 inverted residuals NECK1->NECK2 (first stride2)
    make_invres('mobilevit.encoder.layer.1.layer.0', sd, NECK[1], NECK[2])
    make_invres('mobilevit.encoder.layer.1.layer.1', sd, NECK[2], NECK[2])
    make_invres('mobilevit.encoder.layer.1.layer.2', sd, NECK[2], NECK[2])
    # layer.2..4: MobileViT blocks
    make_mvit_block('mobilevit.encoder.layer.2.', sd, NECK[2], NECK[3],
                    HIDDEN[0], TX_LAYERS[0])
    make_mvit_block('mobilevit.encoder.layer.3.', sd, NECK[3], NECK[4],
                    HIDDEN[1], TX_LAYERS[1])
    make_mvit_block('mobilevit.encoder.layer.4.', sd, NECK[4], NECK[5],
                    HIDDEN[2], TX_LAYERS[2])
    make_convlayer('mobilevit.conv_1x1_exp', sd, NECK[6], NECK[5], 1)
    make_linear('classifier', sd, NUM_CLASSES, NECK[6])
    return sd


# --------------------------------------------------------------------------
# numpy float64 oracle (volumes in (C, H, W); torchvision conv weights
# [O,I,kh,kw]; BN folded into conv).
# --------------------------------------------------------------------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def fold_bn(w, sd, bn):
    bw, bb = sd[bn + '.weight'], sd[bn + '.bias']
    bm, bv = sd[bn + '.running_mean'], sd[bn + '.running_var']
    scale = bw / np.sqrt(bv + BN_EPS)
    return w * scale[:, None, None, None], bb - bw * bm / np.sqrt(bv + BN_EPS)


def conv2d(x, w, b, stride, pad):
    I, H, W = x.shape
    O, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((I, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((O, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            patch = xp[:, iy:iy + k, ix:ix + k]
            out[:, oy, ox] = np.tensordot(w, patch, axes=([1, 2, 3],
                                                          [0, 1, 2])) + b
    return out


def dwconv2d(x, w, b, stride, pad):
    C, H, W = x.shape
    _, _, k, _ = w.shape
    Ho = (H - k + 2 * pad) // stride + 1
    Wo = (W - k + 2 * pad) // stride + 1
    xp = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.zeros((C, Ho, Wo), dtype=np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            iy, ix = oy * stride, ox * stride
            patch = xp[:, iy:iy + k, ix:ix + k]
            out[:, oy, ox] = (w[:, 0] * patch).reshape(C, -1).sum(axis=1) + b
    return out


def convlayer(x, sd, prefix, stride, bn=True, act=True, depthwise=False):
    w = sd[prefix + '.convolution.weight']
    k = w.shape[2]
    pad = (k - 1) // 2
    if bn:
        bw, bb = sd[prefix + '.normalization.weight'], \
            sd[prefix + '.normalization.bias']
        bm, bv = sd[prefix + '.normalization.running_mean'], \
            sd[prefix + '.normalization.running_var']
        scale = bw / np.sqrt(bv + BN_EPS)
        shift = bb - bw * bm / np.sqrt(bv + BN_EPS)
        if depthwise:
            wf = w * scale[:, None, None, None]
            out = dwconv2d(x, wf, shift, stride, pad)
        else:
            wf = w * scale[:, None, None, None]
            out = conv2d(x, wf, shift, stride, pad)
    else:
        zb = np.zeros(w.shape[0])
        out = (dwconv2d if depthwise else conv2d)(x, w, zb, stride, pad)
    if act:
        out = silu(out)
    return out


def invres(x, sd, prefix, stride, in_ch, out_ch):
    # prefix has NO trailing dot.
    exp = make_divisible(in_ch * EXPAND_RATIO, 8)
    has_expand = exp != in_ch
    has_residual = (stride == 1) and (in_ch == out_ch)
    inp = x
    out = x
    if has_expand:
        out = convlayer(out, sd, prefix + '.expand_1x1', 1)
    out = convlayer(out, sd, prefix + '.conv_3x3', stride, depthwise=True)
    out = convlayer(out, sd, prefix + '.reduce_1x1', 1, act=False)
    if has_residual:
        out = out + inp
    return out


def layernorm(x, sd, prefix):
    # x: (..., d) ; LN over last axis
    g, b = sd[prefix + '.weight'], sd[prefix + '.bias']
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * g + b


def linear(x, sd, prefix):
    w = sd[prefix + '.weight']
    out = x @ w.T
    if prefix + '.bias' in sd:
        out = out + sd[prefix + '.bias']
    return out


def attention(x, sd, prefix, hidden):
    # x: (seq, hidden) for ONE sub-position sequence. Multi-head SDPA.
    q = linear(x, sd, prefix + 'attention.attention.query')
    k = linear(x, sd, prefix + 'attention.attention.key')
    v = linear(x, sd, prefix + 'attention.attention.value')
    seq = x.shape[0]
    hd = hidden // NUM_HEADS
    out = np.zeros((seq, hidden), dtype=np.float64)
    for h in range(NUM_HEADS):
        sl = slice(h * hd, (h + 1) * hd)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        scores = qh @ kh.T / math.sqrt(hd)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out[:, sl] = w @ vh
    return linear(out, sd, prefix + 'attention.output.dense')


def txlayer(tokens, sd, prefix, hidden):
    # tokens: (patch_area, num_patches, hidden). Attention is INDEPENDENT per
    # sub-position a (HF puts patch_area on the batch axis).
    pa = tokens.shape[0]
    out = np.empty_like(tokens)
    for a in range(pa):
        x = tokens[a]                              # (num_patches, hidden)
        attn = attention(layernorm(x, sd, prefix + 'layernorm_before'),
                         sd, prefix, hidden)
        x = attn + x
        h = layernorm(x, sd, prefix + 'layernorm_after')
        h = silu(linear(h, sd, prefix + 'intermediate.dense'))
        h = linear(h, sd, prefix + 'output.dense')
        out[a] = h + x
    return out


def unfold(feat, patch):
    # feat: (C, H, W) -> tokens (patch_area, num_patches, C).
    C, H, W = feat.shape
    nph, npw = H // patch, W // patch
    num_patches = nph * npw
    pa = patch * patch
    tokens = np.zeros((pa, num_patches, C), dtype=np.float64)
    for y in range(H):
        ph, dy = y // patch, y % patch
        for xx in range(W):
            pw, dx = xx // patch, xx % patch
            p = ph * npw + pw
            a = dy * patch + dx
            tokens[a, p, :] = feat[:, y, xx]
    return tokens, (C, H, W)


def fold(tokens, info, patch):
    C, H, W = info
    npw = W // patch
    feat = np.zeros((C, H, W), dtype=np.float64)
    for y in range(H):
        ph, dy = y // patch, y % patch
        for xx in range(W):
            pw, dx = xx // patch, xx % patch
            p = ph * npw + pw
            a = dy * patch + dx
            feat[:, y, xx] = tokens[a, p, :]
    return feat


def mvit_block(x, sd, prefix, in_ch, out_ch, hidden, ntx):
    # prefix HAS a trailing dot.
    x = invres(x, sd, prefix + 'downsampling_layer', 2, in_ch, out_ch)
    residual = x
    x = convlayer(x, sd, prefix + 'conv_kxk', 1)
    x = convlayer(x, sd, prefix + 'conv_1x1', 1, bn=False, act=False)
    tokens, info = unfold(x, PATCH)
    for j in range(ntx):
        tokens = txlayer(tokens, sd, prefix + f'transformer.layer.{j}.', hidden)
    tokens = layernorm(tokens, sd, prefix + 'layernorm')
    x = fold(tokens, info, PATCH)
    x = convlayer(x, sd, prefix + 'conv_projection', 1)
    x = np.concatenate([residual, x], axis=0)     # concat along channel
    x = convlayer(x, sd, prefix + 'fusion', 1)
    return x


def forward(pixels, sd):
    x = convlayer(pixels, sd, 'mobilevit.conv_stem', 2)
    x = invres(x, sd, 'mobilevit.encoder.layer.0.layer.0', 1, NECK[0], NECK[1])
    x = invres(x, sd, 'mobilevit.encoder.layer.1.layer.0', 2, NECK[1], NECK[2])
    x = invres(x, sd, 'mobilevit.encoder.layer.1.layer.1', 1, NECK[2], NECK[2])
    x = invres(x, sd, 'mobilevit.encoder.layer.1.layer.2', 1, NECK[2], NECK[2])
    x = mvit_block(x, sd, 'mobilevit.encoder.layer.2.', NECK[2], NECK[3],
                   HIDDEN[0], TX_LAYERS[0])
    x = mvit_block(x, sd, 'mobilevit.encoder.layer.3.', NECK[3], NECK[4],
                   HIDDEN[1], TX_LAYERS[1])
    x = mvit_block(x, sd, 'mobilevit.encoder.layer.4.', NECK[4], NECK[5],
                   HIDDEN[2], TX_LAYERS[2])
    x = convlayer(x, sd, 'mobilevit.conv_1x1_exp', 1)
    pooled = x.reshape(x.shape[0], -1).mean(axis=1)
    logits = sd['classifier.weight'] @ pooled + sd['classifier.bias']
    return logits, x.shape


def main():
    sd = build_state_dict()
    pixels = np.zeros((NUM_CHANNELS, IMAGE, IMAGE), dtype=np.float64)
    for c in range(NUM_CHANNELS):
        for y in range(IMAGE):
            for xx in range(IMAGE):
                pixels[c, y, xx] = (((c * IMAGE * IMAGE + y * IMAGE + xx) * 5)
                                    % 17 - 8) / 8.0

    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()
              if not k.endswith('num_batches_tracked')}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    logits, feat_shape = forward(pixels, sd)
    print(f'feature map before head pool: {feat_shape}, logits {logits.shape}')
    print(f'logits: {logits.tolist()}')

    save_file(sd_f32, 'tests/fixtures/tiny_mobilevit.safetensors')

    config = {
        'model_type': 'mobilevit',
        'image_size': IMAGE,
        'num_channels': NUM_CHANNELS,
        'num_labels': NUM_CLASSES,
        'patch_size': PATCH,
        'num_attention_heads': NUM_HEADS,
        'mlp_ratio': MLP_RATIO,
        'expand_ratio': EXPAND_RATIO,
        'conv_kernel_size': CONV_K,
        'layer_norm_eps': LN_EPS,
        'bn_eps': BN_EPS,
        'qkv_bias': True,
        'hidden_sizes': HIDDEN,
        'neck_hidden_sizes': NECK,
        'transformer_layers': TX_LAYERS,
    }
    with open('tests/fixtures/tiny_mobilevit_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_mobilevit_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels.tolist(),
            'logits': [logits.tolist()],
            'num_labels': NUM_CLASSES,
        }, f)
    print(f'wrote tiny_mobilevit.safetensors ({len(sd_f32)} tensors) + '
          f'config + oracle')

    # ---- fixture self-checks: each major branch must move the logits ----
    base = logits.copy()

    def effect(key):
        alt = dict(sd)
        alt[key] = np.zeros_like(sd[key])
        return np.abs(forward(pixels, alt)[0] - base).max()

    TH = 1e-6
    for key in ['mobilevit.conv_stem.normalization.weight',
                'mobilevit.encoder.layer.1.layer.0.expand_1x1.convolution.weight',
                'mobilevit.encoder.layer.2.conv_1x1.convolution.weight',
                'mobilevit.encoder.layer.2.transformer.layer.0.attention.attention.query.weight',
                'mobilevit.encoder.layer.2.transformer.layer.0.intermediate.dense.weight',
                'mobilevit.encoder.layer.2.layernorm.weight',
                'mobilevit.encoder.layer.2.fusion.convolution.weight',
                'mobilevit.encoder.layer.4.transformer.layer.0.output.dense.weight',
                'mobilevit.conv_1x1_exp.convolution.weight',
                'classifier.bias']:
        d = effect(key)
        assert d > TH, f'{key} had no effect ({d})'
        print(f'  effect {key}: {d:.6f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
