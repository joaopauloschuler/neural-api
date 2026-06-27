#!/usr/bin/env python3
"""Generate a tiny RANDOM h94/IP-Adapter parity fixture for
tests/TestNeuralPretrained.pas (TestIPAdapterParity).

diffusers is NOT installed in the reusable venv, so the reference forward is a
self-contained numpy float64 oracle that mirrors the CAI importer's forward
path EXACTLY (the make_pico recipe). The weights use the exact h94 IP-Adapter +
diffusers ImageProjModel key scheme so the importer is exercised on a real key
layout.

IP-Adapter (https://github.com/tencent-ailab/IP-Adapter, h94/IP-Adapter
ip-adapter_sd15) conditions a FROZEN SD/SDXL UNet on a PROMPT IMAGE via
DECOUPLED cross-attention. This is a DISTINCT mechanism from ControlNet
(spatial feature injection) and T2I-Adapter. Two genuinely-new pieces (and the
ONLY pieces this v1 fixture / parity test isolate and verify):

  (1) image_proj = diffusers ImageProjModel: from the POOLED CLIP image
      embedding (dim = clip_embeddings_dim) a single Linear `proj` (WITH bias)
      maps to clip_extra_context_tokens * cross_attention_dim, reshaped to N
      image tokens of dim cross_attention_dim, then a LayerNorm `norm`. Keys:
          image_proj.proj.weight   (N*cross, clip_dim)
          image_proj.proj.bias     (N*cross,)
          image_proj.norm.weight   (cross,)
          image_proj.norm.bias     (cross,)

  (2) the DECOUPLED cross-attention output of ONE UNet cross-attn block:
          out = Attn(Q, K_txt, V_txt) + scale * Attn(Q, K_img, V_img)
      Q is the SHARED UNet to_q over the unet hidden state (SAME projection the
      base UNet text cross-attn uses); K_txt/V_txt are the base UNet
      attn2.to_k / attn2.to_v over the TEXT states; K_img/V_img are the EXTRA
      bias-free ip-adapter projections to_k_ip / to_v_ip over the N image
      tokens. The two attentions are computed per-head and summed (image stream
      weighted by the fixed conditioning_scale). out is then run through the
      shared to_out.0. Keys (ip_adapter.* indexed by attn-processor order; for
      the isolated block this fixture uses index 1, matching the real
      checkpoint's `1.to_k_ip.weight` style):
          <base UNet attn2>.to_q.weight     (channels, channels)   bias-free
          <base UNet attn2>.to_k.weight     (channels, cross)      bias-free
          <base UNet attn2>.to_v.weight     (channels, cross)      bias-free
          <base UNet attn2>.to_out.0.weight (channels, channels)   biased
          <base UNet attn2>.to_out.0.bias   (channels,)
          ip_adapter.1.to_k_ip.weight       (channels, cross)      bias-free
          ip_adapter.1.to_v_ip.weight       (channels, cross)      bias-free

v1 scope: the plain (non-Plus) ip-adapter_sd15. The fixture pins ONE isolated
decoupled cross-attn block + image_proj; FULL per-block tapping into BuildSDUNet
is a follow-up (the new code -- image_proj loader + the second-K/V cross-attn
path -- is fully exercised by this isolated block). Follow-ups: IP-Adapter-Plus,
FaceID, SDXL key scheme.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/ip_adapter_tiny_fixture.py
writes tests/fixtures/tiny_ip_adapter{.safetensors,_config.json,_io.json}.
Needs numpy + safetensors + scipy(erf, optional) only.
"""
import json
import math
import numpy as np
from safetensors.numpy import save_file

LN_EPS = 1e-5

# ---- pico config ----
CLIP_DIM = 10        # pooled CLIP image-embedding dim (clip_embeddings_dim)
CROSS_DIM = 6        # cross_attention_dim (SD15 text encoder width, pico)
N_TOKENS = 4         # clip_extra_context_tokens (4 for the standard adapter)
CHANNELS = 8         # UNet cross-attn block hidden dim
HEADS = 2            # attention heads
HW = 9               # UNet hidden state token count (3x3 spatial grid flattened)
TEXT_SEQ = 5         # text encoder_hidden_states length
SCALE = 0.7          # conditioning_scale (fixed)
ATTN_IDX = 1         # ip_adapter.<idx>.* attn-processor index (matches h94 keys)

rng = np.random.default_rng(20260627)


def randn(*shape, std=1.0):
    return (rng.standard_normal(shape) * std).astype(np.float64)


def lin_w(out_f, in_f, std=0.12):
    return randn(out_f, in_f, std=std)


# ===========================================================================
# numpy float64 oracle.
# ===========================================================================
def layer_norm(x, gamma, beta, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def image_proj(image_embeds, sd):
    """ImageProjModel: pooled image_embeds (CLIP_DIM,) -> N image tokens
    (N_TOKENS, CROSS_DIM)."""
    h = image_embeds @ sd['image_proj.proj.weight'].T + sd['image_proj.proj.bias']
    tokens = h.reshape(N_TOKENS, CROSS_DIM)
    tokens = layer_norm(tokens, sd['image_proj.norm.weight'],
                        sd['image_proj.norm.bias'], LN_EPS)
    return tokens


def mh_attention(q, k, v, heads):
    """Multi-head scaled-dot-product attention. q (Nq,C), k/v (Nk,C)."""
    c = q.shape[1]
    head_dim = c // heads
    scale = 1.0 / math.sqrt(head_dim)
    Nq = q.shape[0]
    out = np.zeros((Nq, c), dtype=np.float64)
    for hh in range(heads):
        sl = slice(hh * head_dim, (hh + 1) * head_dim)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        scores = (qh @ kh.T) * scale
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=1, keepdims=True)
        out[:, sl] = w @ vh
    return out


def decoupled_cross_attention(hidden, text, img_tokens, sd, attn_prefix):
    """out = Attn(Q,K_txt,V_txt) + scale*Attn(Q,K_img,V_img), then to_out.0.

    hidden (HW,CHANNELS) is the UNet hidden state (Q source); text
    (TEXT_SEQ,CROSS_DIM); img_tokens (N_TOKENS,CROSS_DIM)."""
    q = hidden @ sd[attn_prefix + 'to_q.weight'].T          # shared Q
    k_txt = text @ sd[attn_prefix + 'to_k.weight'].T
    v_txt = text @ sd[attn_prefix + 'to_v.weight'].T
    k_img = img_tokens @ sd[f'ip_adapter.{ATTN_IDX}.to_k_ip.weight'].T
    v_img = img_tokens @ sd[f'ip_adapter.{ATTN_IDX}.to_v_ip.weight'].T
    out_txt = mh_attention(q, k_txt, v_txt, HEADS)
    out_img = mh_attention(q, k_img, v_img, HEADS)
    out = out_txt + SCALE * out_img
    out = out @ sd[attn_prefix + 'to_out.0.weight'].T + sd[attn_prefix + 'to_out.0.bias']
    return out


# ===========================================================================
# State dict (exact h94 IP-Adapter + diffusers attn2 key scheme).
# ===========================================================================
def build_state_dict():
    sd = {}
    # ---- image_proj (diffusers ImageProjModel) ----
    sd['image_proj.proj.weight'] = lin_w(N_TOKENS * CROSS_DIM, CLIP_DIM)
    sd['image_proj.proj.bias'] = randn(N_TOKENS * CROSS_DIM, std=0.1)
    sd['image_proj.norm.weight'] = randn(CROSS_DIM, std=0.3) + 1.0
    sd['image_proj.norm.bias'] = randn(CROSS_DIM, std=0.25)
    # ---- base UNet cross-attn (attn2) projections (the shared Q + text K/V +
    #      out). These come from the FROZEN base UNet checkpoint; the fixture
    #      carries them under the diffusers attn2 key scheme so the importer can
    #      read the SAME tensors the SD UNet importer reads. ----
    ap = 'unet.attn2.'
    sd[ap + 'to_q.weight'] = lin_w(CHANNELS, CHANNELS)
    sd[ap + 'to_k.weight'] = lin_w(CHANNELS, CROSS_DIM)
    sd[ap + 'to_v.weight'] = lin_w(CHANNELS, CROSS_DIM)
    sd[ap + 'to_out.0.weight'] = lin_w(CHANNELS, CHANNELS)
    sd[ap + 'to_out.0.bias'] = randn(CHANNELS, std=0.1)
    # ---- ip_adapter extra K/V (bias-free) ----
    sd[f'ip_adapter.{ATTN_IDX}.to_k_ip.weight'] = lin_w(CHANNELS, CROSS_DIM)
    sd[f'ip_adapter.{ATTN_IDX}.to_v_ip.weight'] = lin_w(CHANNELS, CROSS_DIM)
    return sd, ap


def main():
    sd, ap = build_state_dict()
    sd_f32 = {k: v.astype(np.float32) for k, v in sd.items()}
    sd = {k: v.astype(np.float64) for k, v in sd_f32.items()}

    # deterministic inputs.
    image_embeds = np.zeros(CLIP_DIM, dtype=np.float64)
    for d in range(CLIP_DIM):
        image_embeds[d] = (((d * 7) % 13) - 6) / 8.0
    hidden = np.zeros((HW, CHANNELS), dtype=np.float64)
    for s in range(HW):
        for d in range(CHANNELS):
            hidden[s, d] = (((s * 16 + d) * 5) % 11 - 5) / 8.0
    text = np.zeros((TEXT_SEQ, CROSS_DIM), dtype=np.float64)
    for s in range(TEXT_SEQ):
        for d in range(CROSS_DIM):
            text[s, d] = (((s * 13 + d) * 3) % 17 - 8) / 16.0

    img_tokens = image_proj(image_embeds, sd)
    out = decoupled_cross_attention(hidden, text, img_tokens, sd, ap)
    print(f'image_embeds {image_embeds.shape} hidden {hidden.shape} '
          f'text {text.shape}')
    print(f'image tokens {img_tokens.shape} cross-attn out {out.shape}')

    save_file(sd_f32, 'tests/fixtures/tiny_ip_adapter.safetensors')
    config = {
        '_class_name': 'IPAdapterModel',
        'clip_embeddings_dim': CLIP_DIM,
        'cross_attention_dim': CROSS_DIM,
        'clip_extra_context_tokens': N_TOKENS,
        'channels': CHANNELS,
        'num_heads': HEADS,
        'hidden_tokens': HW,
        'text_seq_len': TEXT_SEQ,
        'conditioning_scale': SCALE,
        'attn_index': ATTN_IDX,
    }
    with open('tests/fixtures/tiny_ip_adapter_config.json', 'w') as f:
        json.dump(config, f, indent=1)
    with open('tests/fixtures/tiny_ip_adapter_io.json', 'w') as f:
        json.dump({
            'image_embeds': image_embeds.tolist(),
            'hidden_state': hidden.tolist(),
            'text_states': text.tolist(),
            'image_tokens': img_tokens.tolist(),
            'cross_attn_output': out.tolist(),
        }, f)
    print(f'wrote tiny_ip_adapter.safetensors ({len(sd_f32)} tensors) + config + io')

    # ---- fixture self-checks: every NEW piece must MATTER. ----
    base = out

    def maxdiff(o):
        return np.abs(o - base).max()

    # image stream matters: zeroing scale changes the output by exactly
    # scale*out_img.
    def cross(scale_override=None, mut=None):
        s = dict(sd)
        if mut:
            s.update(mut)
        if scale_override is None:
            return decoupled_cross_attention(hidden, text, img_tokens, s, ap)
        # recompute with a different scale by temporarily swapping the global.
        global SCALE
        old = SCALE
        SCALE = scale_override
        try:
            return decoupled_cross_attention(hidden, text, img_tokens, s, ap)
        finally:
            SCALE = old

    d = maxdiff(cross(scale_override=0.0))
    assert d > 1e-4, f'image stream no effect ({d})'
    print(f'image-stream (scale) effect: {d:.4f}')

    # image embeds matter (through image_proj).
    alt_tokens = image_proj(image_embeds + 0.5, sd)
    d = np.abs(decoupled_cross_attention(hidden, text, alt_tokens, sd, ap) - base).max()
    assert d > 1e-4, f'image embeds no effect ({d})'
    print(f'image-embeds effect: {d:.4f}')

    # to_k_ip matters.
    d = maxdiff(cross(mut={f'ip_adapter.{ATTN_IDX}.to_k_ip.weight':
                           sd[f'ip_adapter.{ATTN_IDX}.to_k_ip.weight'] + 0.3}))
    assert d > 1e-4, f'to_k_ip no effect ({d})'
    print(f'to_k_ip effect: {d:.4f}')

    # to_v_ip matters.
    d = maxdiff(cross(mut={f'ip_adapter.{ATTN_IDX}.to_v_ip.weight':
                           sd[f'ip_adapter.{ATTN_IDX}.to_v_ip.weight'] + 0.3}))
    assert d > 1e-4, f'to_v_ip no effect ({d})'
    print(f'to_v_ip effect: {d:.4f}')

    # image_proj.norm matters.
    d = np.abs(decoupled_cross_attention(
        hidden, text,
        image_proj(image_embeds,
                   {**sd, 'image_proj.norm.bias': sd['image_proj.norm.bias'] + 0.5}),
        sd, ap) - base).max()
    assert d > 1e-4, f'image_proj.norm no effect ({d})'
    print(f'image_proj.norm effect: {d:.4f}')

    # text stream still matters (shared Q path is real).
    d = np.abs(decoupled_cross_attention(hidden, text + 0.4, img_tokens, sd, ap) - base).max()
    assert d > 1e-4, f'text no effect ({d})'
    print(f'text-stream effect: {d:.4f}')
    print('all fixture self-checks passed')


if __name__ == '__main__':
    main()
