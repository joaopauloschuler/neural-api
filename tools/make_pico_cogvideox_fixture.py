#!/usr/bin/env python3
"""Generate a tiny RANDOM CogVideoX (THUDM/CogVideoX-2b) native text-to-VIDEO
parity fixture for tests/TestNeuralPretrained.pas.

No network access. The denoiser reference is a self-contained float64 numpy
re-implementation of the canonical diffusers CogVideoXTransformer3DModel block;
the VAE reference runs diffusers AutoencoderKLCogVideoX itself (needs torch +
diffusers, e.g. the ~/x venv)
(Yang et al. 2024, "CogVideoX: Text-to-Video Diffusion Models with An Expert
Transformer", arXiv:2408.06072; diffusers.models.transformers.cogvideox_transformer_3d
and diffusers.models.autoencoders.autoencoder_kl_cogvideox).

Two parity targets are written, each asserted < 1e-4 by the Pascal test:

  (A) ONE DENOISER STEP -- the flat MMDiT-style transformer over a flattened
      (frame x height x width) video-latent token sequence, with T5 text
      conditioning and expert adaLN-Zero modulation.  Architecture (pico, one
      block, faithful to diffusers CogVideoXBlock):

        c      = SiLU(time_embed(t))                         # cond vector, width d
        # text states are projected once into the joint width:
        txt    = silu? no -- linear text_proj(text_states)   # (Lt, d)
        # video latent (T,H,W,Cin) is flattened to (Lv=T*H*W, Cin) and patch-
        # embedded by a per-cell Linear (patch_size_t = patch_size = 1 in pico):
        vid    = linear(patch_embed, flat_latent)            # (Lv, d)
        x      = concat([txt, vid])                          # (Lt+Lv, d) joint seq

        # ---- CogVideoXBlock ----
        # norm1 = CogVideoXLayerNormZero: ONE Linear(SiLU(c)) -> 6*d, split into
        #   (shift, scale, gate) for the HIDDEN (video) stream and the SAME
        #   triple reused (CogVideoX shares norm across the joint seq via a single
        #   LayerNorm then applies the hidden gate to video and encoder gate to
        #   text -- diffusers CogVideoXLayerNormZero returns gate, enc_gate).
        # Here, matching diffusers, norm1.linear -> 6*d gives
        #   (shift, scale, gate, enc_shift, enc_scale, enc_gate).
        ns, sc, g, ens, ensc, eng = chunk(linear(silu(c), norm1.w, norm1.b), 6)
        nx_v = modulate(LN(vid), ns, sc)      # video normed+modulated
        nx_t = modulate(LN(txt), ens, ensc)   # text  normed+modulated
        h    = concat([nx_t, nx_v])           # joint normed sequence (txt first)
        # attention: fused q/k/v over the JOINT sequence; 3D RoPE applied to the
        # VIDEO portion of q and k only (text tokens get NO rotary).
        q,k,v = to_q(h), to_k(h), to_v(h)
        q,k   = apply_3d_rope_to_video_part(q,k)
        o     = softmax(q k^T / sqrt(head_dim)) v   # full (non-causal) attention
        o     = to_out(o)
        # gated residual: video gets g, text gets enc_gate.
        vid = vid + g  * o[Lt:]
        txt = txt + eng* o[:Lt]
        # norm2 + FFN (CogVideoXLayerNormZero again -> 6*d), gated residual.
        ns2, sc2, g2, ens2, ensc2, eng2 = chunk(linear(silu(c), norm2.w, norm2.b),6)
        fx_v = modulate(LN(vid), ns2, sc2)
        fx_t = modulate(LN(txt), ens2, ensc2)
        hh   = concat([fx_t, fx_v])
        m    = ff2(gelu_tanh(ff1(hh)))
        vid  = vid + g2 * m[Lt:]
        txt  = txt + eng2* m[:Lt]
        # ---- final layer: norm_final + adaLN(SiLU(c)->2*d) + proj_out ----
        ns_f, sc_f = chunk(linear(silu(c), norm_out.w, norm_out.b), 2)
        out_v = modulate(LN(vid), ns_f, sc_f)
        eps   = linear(proj_out, out_v)        # (Lv, Cout) raw eps prediction
      The denoiser output is the (T,H,W,Cout) raw eps; the Pascal side reshapes.

      3D RoPE: head_dim is split into three equal sections (dt, dh, dw) of pairs;
      section t rotates by the frame index, h by the row, w by the column -- the
      diffusers get_3d_rotary_pos_embed factored layout, expressed by the repo's
      TNNetMRotaryEmbedding (mrope_section = dt,dh,dw pairs).

  (B) THE FULL AutoencoderKLCogVideoX DECODER -- a pico diffusers
      AutoencoderKLCogVideoX (random weights, real key names, written with
      save_pretrained into tests/fixtures/tiny_cogvideox_vae*/) and a float64
      reference vae.decode(latent / scaling_factor) computed by diffusers
      itself: conv_in, the mid block, the up blocks with their temporal 2x
      upsample (first frame kept apart on an odd frame count), norm_out
      (SpatialNorm3D on the latent), SiLU and conv_out.  Two VAEs: the main one
      (4 up blocks, 2 of them temporal, 3 latent frames, no post_quant_conv)
      and a 2-block one with post_quant_conv over 2 latent frames (diffusers
      sizes post_quant_conv by out_channels, so it needs out == latent).
      With <= 3 latent frames diffusers decodes the whole clip in one batch;
      the "vae_batched" cases (4, 5 and 7 latent frames) go through its
      frame-batched _decode (2 latent frames per batch, the odd one in the
      first, each causal conv carrying the last 2 input frames of the batch
      before it).

The fixture writes RAW diffusers-style tensor names; the Pascal importer reads
exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_cogvideox_fixture.py
writes tests/fixtures/tiny_cogvideox{.safetensors,_config.json,_io.json},
tests/fixtures/tiny_cogvideox_vae{,_pq}/ and tiny_cogvideox_vae_io.json.
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file

# ---------------- pico config ----------------
HIDDEN = 16
HEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 8 -> 4 channel-pairs
MLP_HIDDEN = HIDDEN * 4             # 64
T = 2                              # latent frames (>=2 to exercise temporal causality)
GH = 2                             # latent grid height
GW = 2                             # latent grid width
LV = T * GH * GW                  # video tokens = 8
LT = 3                            # text tokens
CIN = 4                           # latent in-channels (denoiser input)
COUT = 4                          # denoiser eps out-channels
TEXT_DIM = 6                      # T5 text-encoder width
ROPE_BASE = 10000.0
# 3D RoPE section split over the HEAD_DIM//2 = 4 channel-pairs: (t,h,w)
SEC_T, SEC_H, SEC_W = 2, 1, 1     # sums to 4 = HEAD_DIM//2

RNG = np.random.default_rng(20260626)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# ---------------- math helpers (float64) ----------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layernorm_noaffine(x, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def linear(x, w, b):
    return x @ w.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def modulate(h, shift, scale):
    return h * (1.0 + scale) + shift


def time_sinusoid(t, dim):
    # diffusers get_timestep_embedding, flip_sin_to_cos=True, downscale_freq_shift=0
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = t * freqs
    return np.concatenate([np.cos(args), np.sin(args)])  # [cos|sin]


# ---------------- weights ----------------
W = {}
# time embedding MLP (256-d sinusoid -> d -> d)
TIME_SIN = 256
W["time_embedding.linear_1.weight"] = randn(HIDDEN, TIME_SIN)
W["time_embedding.linear_1.bias"] = randn(HIDDEN)
W["time_embedding.linear_2.weight"] = randn(HIDDEN, HIDDEN)
W["time_embedding.linear_2.bias"] = randn(HIDDEN)
# text projection (T5 width -> d)
W["text_proj.weight"] = randn(HIDDEN, TEXT_DIM)
W["text_proj.bias"] = randn(HIDDEN)
# video patch embed (Cin -> d), per cell (patch_t = patch = 1 in pico)
W["patch_embed.weight"] = randn(HIDDEN, CIN)
W["patch_embed.bias"] = randn(HIDDEN)
# block norm1 (CogVideoXLayerNormZero) -> 6*d
W["transformer_blocks.0.norm1.linear.weight"] = randn(6 * HIDDEN, HIDDEN)
W["transformer_blocks.0.norm1.linear.bias"] = randn(6 * HIDDEN)
# block attention q/k/v + out
W["transformer_blocks.0.attn1.to_q.weight"] = randn(HIDDEN, HIDDEN)
W["transformer_blocks.0.attn1.to_q.bias"] = randn(HIDDEN)
W["transformer_blocks.0.attn1.to_k.weight"] = randn(HIDDEN, HIDDEN)
W["transformer_blocks.0.attn1.to_k.bias"] = randn(HIDDEN)
W["transformer_blocks.0.attn1.to_v.weight"] = randn(HIDDEN, HIDDEN)
W["transformer_blocks.0.attn1.to_v.bias"] = randn(HIDDEN)
W["transformer_blocks.0.attn1.to_out.0.weight"] = randn(HIDDEN, HIDDEN)
W["transformer_blocks.0.attn1.to_out.0.bias"] = randn(HIDDEN)
# block norm2 -> 6*d
W["transformer_blocks.0.norm2.linear.weight"] = randn(6 * HIDDEN, HIDDEN)
W["transformer_blocks.0.norm2.linear.bias"] = randn(6 * HIDDEN)
# block FFN
W["transformer_blocks.0.ff.net.0.proj.weight"] = randn(MLP_HIDDEN, HIDDEN)
W["transformer_blocks.0.ff.net.0.proj.bias"] = randn(MLP_HIDDEN)
W["transformer_blocks.0.ff.net.2.weight"] = randn(HIDDEN, MLP_HIDDEN)
W["transformer_blocks.0.ff.net.2.bias"] = randn(HIDDEN)
# final layer
W["norm_out.linear.weight"] = randn(2 * HIDDEN, HIDDEN)
W["norm_out.linear.bias"] = randn(2 * HIDDEN)
W["proj_out.weight"] = randn(COUT, HIDDEN)
W["proj_out.bias"] = randn(COUT)


# ---------------- 3D RoPE ----------------
def build_3d_rope():
    # returns cos, sin each (LV, HEAD_DIM); text tokens get identity (handled
    # by only rotating the video part).
    half = HEAD_DIM // 2  # 4 pairs
    inv_freq = ROPE_BASE ** (-np.arange(half) / half)  # per-pair base freq
    cos = np.zeros((LV, HEAD_DIM))
    sin = np.zeros((LV, HEAD_DIM))
    sec = [0] * SEC_T + [1] * SEC_H + [2] * SEC_W  # which axis each pair uses
    idx = 0
    for ft in range(T):
        for hh in range(GH):
            for ww in range(GW):
                pos = (ft, hh, ww)
                for p in range(half):
                    axis = sec[p]
                    ang = pos[axis] * inv_freq[p]
                    cos[idx, 2 * p] = np.cos(ang)
                    cos[idx, 2 * p + 1] = np.cos(ang)
                    sin[idx, 2 * p] = np.sin(ang)
                    sin[idx, 2 * p + 1] = np.sin(ang)
                idx += 1
    return cos, sin


def apply_rope_pairs(x, cos, sin):
    # x: (LV, HEAD_DIM); rotate each adjacent (2p,2p+1) pair:
    #   x0' = x0*cos - x1*sin ; x1' = x1*cos + x0*sin
    out = np.empty_like(x)
    half = HEAD_DIM // 2
    for p in range(half):
        a = x[:, 2 * p]
        b = x[:, 2 * p + 1]
        out[:, 2 * p] = a * cos[:, 2 * p] - b * sin[:, 2 * p]
        out[:, 2 * p + 1] = b * cos[:, 2 * p + 1] + a * sin[:, 2 * p + 1]
    return out


COS3D, SIN3D = build_3d_rope()


def heads_split(x):
    n = x.shape[0]
    return x.reshape(n, HEADS, HEAD_DIM).transpose(1, 0, 2)


def heads_merge(x):
    n = x.shape[1]
    return x.transpose(1, 0, 2).reshape(n, HIDDEN)


def denoiser_forward(latent, t, text_states):
    # latent: (T,GH,GW,CIN) ; text_states: (LT, TEXT_DIM)
    c = silu(linear(time_sinusoid(t, TIME_SIN),
                    W["time_embedding.linear_1.weight"],
                    W["time_embedding.linear_1.bias"]))
    c = linear(c, W["time_embedding.linear_2.weight"],
               W["time_embedding.linear_2.bias"])
    c = silu(c)  # cond vector fed to the adaLN linears
    txt = linear(text_states, W["text_proj.weight"], W["text_proj.bias"])  # (LT,d)
    flat = latent.reshape(LV, CIN)
    vid = linear(flat, W["patch_embed.weight"], W["patch_embed.bias"])     # (LV,d)

    def adaln(vidx, txtx, pfx):
        mod = linear(c, W[pfx + ".linear.weight"], W[pfx + ".linear.bias"])
        ns, sc, g, ens, ensc, eng = np.split(mod, 6)
        nv = modulate(layernorm_noaffine(vidx), ns, sc)
        nt = modulate(layernorm_noaffine(txtx), ens, ensc)
        return nv, nt, g, eng

    # ---- attention sub-block ----
    nv, nt, g, eng = adaln(vid, txt, "transformer_blocks.0.norm1")
    h = np.concatenate([nt, nv], axis=0)  # (LT+LV, d), text first
    q = linear(h, W["transformer_blocks.0.attn1.to_q.weight"],
               W["transformer_blocks.0.attn1.to_q.bias"])
    k = linear(h, W["transformer_blocks.0.attn1.to_k.weight"],
               W["transformer_blocks.0.attn1.to_k.bias"])
    v = linear(h, W["transformer_blocks.0.attn1.to_v.weight"],
               W["transformer_blocks.0.attn1.to_v.bias"])
    qh, kh, vh = heads_split(q), heads_split(k), heads_split(v)
    # 3D RoPE on the VIDEO portion (tokens LT..LT+LV) of q,k per head.
    for hd in range(HEADS):
        qh[hd, LT:, :] = apply_rope_pairs(qh[hd, LT:, :], COS3D, SIN3D)
        kh[hd, LT:, :] = apply_rope_pairs(kh[hd, LT:, :], COS3D, SIN3D)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = softmax((qh @ kh.transpose(0, 2, 1)) * scale, axis=-1)
    o = heads_merge(attn @ vh)
    o = linear(o, W["transformer_blocks.0.attn1.to_out.0.weight"],
               W["transformer_blocks.0.attn1.to_out.0.bias"])
    vid = vid + g * o[LT:]
    txt = txt + eng * o[:LT]

    # ---- FFN sub-block ----
    nv, nt, g2, eng2 = adaln(vid, txt, "transformer_blocks.0.norm2")
    hh = np.concatenate([nt, nv], axis=0)
    m = linear(gelu_tanh(linear(hh, W["transformer_blocks.0.ff.net.0.proj.weight"],
                                W["transformer_blocks.0.ff.net.0.proj.bias"])),
               W["transformer_blocks.0.ff.net.2.weight"],
               W["transformer_blocks.0.ff.net.2.bias"])
    vid = vid + g2 * m[LT:]
    txt = txt + eng2 * m[:LT]

    # ---- final layer (video only) ----
    mod = linear(c, W["norm_out.linear.weight"], W["norm_out.linear.bias"])
    ns_f, sc_f = np.split(mod, 2)
    out_v = modulate(layernorm_noaffine(vid), ns_f, sc_f)
    eps = linear(out_v, W["proj_out.weight"], W["proj_out.bias"])  # (LV, COUT)
    return eps.reshape(T, GH, GW, COUT)


# ---------------- run cases ----------------
den_cases = []
for case in range(3):
    latent = randn(T, GH, GW, CIN, scale=1.0)
    t = float(RNG.integers(1, 1000))
    text = randn(LT, TEXT_DIM, scale=1.0)
    eps = denoiser_forward(latent, t, text)
    den_cases.append({
        "latent": latent.reshape(-1).tolist(),
        "t": t,
        "text": text.reshape(-1).tolist(),
        "eps": eps.reshape(-1).tolist(),
    })

# ---------------- AutoencoderKLCogVideoX decoder (diffusers) ----------------
# (folder, block_out_channels, temporal_compression_ratio, out_channels,
#  use_post_quant_conv, scaling_factor, latent frames, latent H, latent W)
VAE_CASES = [
    ("tiny_cogvideox_vae", (4, 4, 8, 8), 4, 3, False, 1.15258426, 3, 2, 3),
    ("tiny_cogvideox_vae_pq", (4, 4), 2, CIN, True, 0.7, 2, 2, 3),
]
# Clips longer than 3 latent frames, decoded by diffusers' frame-batched
# _decode (2 latent frames per batch plus the conv cache) on the VAEs above:
# (folder, latent frames, latent H, latent W).
VAE_BATCHED_CASES = [
    ("tiny_cogvideox_vae", 5, 1, 2),
    ("tiny_cogvideox_vae_pq", 4, 2, 3),
    ("tiny_cogvideox_vae_pq", 7, 2, 3),
]


def vae_io_case(vae, folder, scaling, lt, lh, lw):
    import torch
    # float32-representable latent, so Pascal reads the same input.
    latent = torch.randn(1, CIN, lt, lh, lw).float().double()
    with torch.no_grad():
        decoded = vae.decode(latent / scaling).sample  # (1, C, T, H, W)
        if lt > vae.num_latent_frames_batch_size + 1:
            z = latent / scaling
            if vae.post_quant_conv is not None:
                z = vae.post_quant_conv(z)
            whole, _ = vae.decoder(z)
            print(f"  {folder}: batched vs whole-clip decode max |diff| = "
                  f"{(decoded - whole).abs().max().item():.3g}")
    print(f"  {folder}: latent {lt}x{lh}x{lw} -> decoded "
          f"{tuple(decoded.shape[2:])} x {decoded.shape[1]}")
    # Both stored as flat (t, h, w, c), the denoiser's token-major layout.
    return {
        "folder": folder,
        "latent_frames": lt, "latent_height": lh, "latent_width": lw,
        "latent": latent[0].permute(1, 2, 3, 0).reshape(-1).tolist(),
        "frames": decoded.shape[2], "height": decoded.shape[3],
        "width": decoded.shape[4], "channels": decoded.shape[1],
        "decoded": [float("%.9g" % v) for v in
                    decoded[0].permute(1, 2, 3, 0).reshape(-1).tolist()],
    }


def write_vae_fixtures(fixtures):
    import torch
    from diffusers import AutoencoderKLCogVideoX
    torch.manual_seed(20261023)
    io_cases = []
    vaes = {}
    for (folder, blocks, ratio, out_ch, post_quant, scaling, lt, lh, lw) in VAE_CASES:
        vae = AutoencoderKLCogVideoX(
            in_channels=3, out_channels=out_ch, latent_channels=CIN,
            down_block_types=("CogVideoXDownBlock3D",) * len(blocks),
            up_block_types=("CogVideoXUpBlock3D",) * len(blocks),
            block_out_channels=blocks, layers_per_block=1, norm_num_groups=2,
            temporal_compression_ratio=ratio, scaling_factor=scaling,
            use_post_quant_conv=post_quant)
        with torch.no_grad():
            # Random affine norms and biases; convs keep their default init.
            for name, param in vae.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    param.copy_(1.0 + 0.2 * torch.randn_like(param))
                elif name.endswith("bias"):
                    param.copy_(0.2 * torch.randn_like(param))
        vae.save_pretrained(os.path.join(fixtures, folder))
        vae = vae.double().eval()
        vaes[folder] = (vae, scaling)
        io_cases.append(vae_io_case(vae, folder, scaling, lt, lh, lw))
    batched_cases = []
    for (folder, lt, lh, lw) in VAE_BATCHED_CASES:
        vae, scaling = vaes[folder]
        batched_cases.append(vae_io_case(vae, folder, scaling, lt, lh, lw))
    with open(os.path.join(fixtures, "tiny_cogvideox_vae_io.json"), "w") as f:
        json.dump({"vae": io_cases, "vae_batched": batched_cases}, f)


# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {kk: vv.astype(np.float32) for kk, vv in W.items()}
save_file(tensors, os.path.join(fixtures, "tiny_cogvideox.safetensors"))

config = {
    "_class_name": "CogVideoXTransformer3DModel",
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
    "num_layers": 1,
    "mlp_hidden": MLP_HIDDEN,
    "num_frames": T,
    "grid_height": GH,
    "grid_width": GW,
    "in_channels": CIN,
    "out_channels": COUT,
    "text_dim": TEXT_DIM,
    "text_seq_len": LT,
    "rope_base": ROPE_BASE,
    "rope_section_t": SEC_T,
    "rope_section_h": SEC_H,
    "rope_section_w": SEC_W,
    "layer_norm_eps": 1e-5,
}
with open(os.path.join(fixtures, "tiny_cogvideox_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_cogvideox_io.json"), "w") as f:
    json.dump({"denoiser": den_cases}, f)

print("wrote tiny_cogvideox.safetensors,_config.json,_io.json to", fixtures)
write_vae_fixtures(fixtures)
print(f"  hidden={HIDDEN} heads={HEADS} head_dim={HEAD_DIM} T={T} grid={GH}x{GW} "
      f"Lv={LV} Lt={LT} sec=({SEC_T},{SEC_H},{SEC_W})")
