#!/usr/bin/env python3
"""Generate a tiny RANDOM MMDiT (Stable Diffusion 3 / FLUX.1) joint-attention
block parity fixture for tests/TestNeuralPretrained.pas.

No network access: the block is randomly initialized from a pico config, never
downloaded -- the diffusers package is NOT installed in this environment, so the
reference forward is a self-contained float64 numpy re-implementation of the
canonical diffusers MMDiT JointTransformerBlock (Esser et al. 2024, "Scaling
Rectified Flow Transformers for High-Resolution Image Synthesis",
arXiv:2403.03206; the SD3Transformer2DModel / JointTransformerBlock in
diffusers.models.attention).

The genuinely NEW architectural piece vs the landed class-conditional DiT
(BuildDiTFromSafeTensors) and the single-stream cross-attention PixArt is the
DUAL-STREAM JOINT-ATTENTION block: image tokens and text tokens carry SEPARATE
per-stream adaLN modulations, QKV projections and MLPs, but their Q/K/V are
CONCATENATED along the sequence axis for ONE joint self-attention pass (text and
image attend to each other symmetrically), then the output is split back per
stream and each stream applies its own output projection + gated residual + MLP.
This is NOT the image->text CROSS-attention of PixArt.

Reference math (diffusers JointTransformerBlock, context_pre_only=False, the
standard joint block; qk_norm=None as in stabilityai/stable-diffusion-3-medium):

  c            = conditioning vector (pooled timestep+text embedding), width d
  # image stream adaLN (norm1):   Linear(SiLU(c)) -> 6*d
  (sh_msa,sc_msa,g_msa,sh_mlp,sc_mlp,g_mlp) = chunk(.,6)
  # text  stream adaLN (norm1_context): Linear(SiLU(c)) -> 6*d
  (c_sh_msa,c_sc_msa,c_g_msa,c_sh_mlp,c_sc_mlp,c_g_mlp) = chunk(.,6)

  hi = modulate(LN(img), sh_msa, sc_msa)          # img normed+modulated
  ht = modulate(LN(txt), c_sh_msa, c_sc_msa)      # txt normed+modulated

  # per-stream q/k/v projections (attn.to_{q,k,v} for img; attn.add_{q,k,v}_proj
  # for txt), each d->d biased, split into heads:
  qi,ki,vi = to_q(hi), to_k(hi), to_v(hi)
  qt,kt,vt = add_q(ht), add_k(ht), add_v(ht)
  # CONCATENATE along the sequence axis (diffusers: [img ; txt] in attn order):
  q = cat([qi,qt]); k = cat([ki,kt]); v = cat([vi,vt])
  o = softmax(q k^T / sqrt(head_dim)) v          # ONE joint attention
  oi, ot = split(o, [img_len, txt_len])          # split back per stream
  img = img + g_msa  * to_out(oi)                # img out-proj attn.to_out.0
  txt = txt + c_g_msa* to_add_out(ot)            # txt out-proj attn.to_add_out
  img = img + g_mlp  * mlp_img( modulate(LN(img), sh_mlp, sc_mlp) )
  txt = txt + c_g_mlp* mlp_txt( modulate(LN(txt), c_sh_mlp, c_sc_mlp) )

  modulate(h,shift,scale)=h*(1+scale)+shift; the LNs are elementwise_affine=False
  (no learned gain/bias - the adaLN supplies scale/shift); MLP is
  fc2( gelu_tanh( fc1(.) ) ) (diffusers ff "gelu-approximate").

The fixture writes RAW diffusers JointTransformerBlock tensor names (norm1.*,
norm1_context.*, attn.to_q/to_k/to_v/to_out.0, attn.add_q_proj/add_k_proj/
add_v_proj/to_add_out, ff.net.*, ff_context.net.*); the Pascal importer reads
exactly these.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_mmdit_fixture.py
writes tests/fixtures/tiny_mmdit{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
from safetensors.numpy import save_file

# ---------------- pico config ----------------
HIDDEN = 16
HEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 8
MLP_HIDDEN = HIDDEN * 4             # 64 (diffusers ff mult=4)
IMG_LEN = 5                         # image tokens in this block
TXT_LEN = 3                         # text tokens in this block

RNG = np.random.default_rng(20260615)


def randn(*shape, scale=0.3):
    return (RNG.standard_normal(shape) * scale).astype(np.float64)


# ---------------- math helpers (float64) ----------------
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layernorm_noaffine(x, eps=1e-6):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def linear(x, w, b):
    # w: (out, in) torch convention, b: (out,)
    return x @ w.T + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def modulate(h, shift, scale):
    return h * (1.0 + scale) + shift


# ---------------- weights ----------------
W = {}
# image-stream adaLN modulation (norm1.linear): Linear(d -> 6*d)
W["norm1.linear.weight"] = randn(6 * HIDDEN, HIDDEN)
W["norm1.linear.bias"] = randn(6 * HIDDEN)
# text-stream adaLN modulation (norm1_context.linear): Linear(d -> 6*d)
W["norm1_context.linear.weight"] = randn(6 * HIDDEN, HIDDEN)
W["norm1_context.linear.bias"] = randn(6 * HIDDEN)
# joint attention: image-stream q/k/v + out-proj
W["attn.to_q.weight"] = randn(HIDDEN, HIDDEN)
W["attn.to_q.bias"] = randn(HIDDEN)
W["attn.to_k.weight"] = randn(HIDDEN, HIDDEN)
W["attn.to_k.bias"] = randn(HIDDEN)
W["attn.to_v.weight"] = randn(HIDDEN, HIDDEN)
W["attn.to_v.bias"] = randn(HIDDEN)
W["attn.to_out.0.weight"] = randn(HIDDEN, HIDDEN)
W["attn.to_out.0.bias"] = randn(HIDDEN)
# joint attention: text-stream q/k/v + out-proj
W["attn.add_q_proj.weight"] = randn(HIDDEN, HIDDEN)
W["attn.add_q_proj.bias"] = randn(HIDDEN)
W["attn.add_k_proj.weight"] = randn(HIDDEN, HIDDEN)
W["attn.add_k_proj.bias"] = randn(HIDDEN)
W["attn.add_v_proj.weight"] = randn(HIDDEN, HIDDEN)
W["attn.add_v_proj.bias"] = randn(HIDDEN)
W["attn.to_add_out.weight"] = randn(HIDDEN, HIDDEN)
W["attn.to_add_out.bias"] = randn(HIDDEN)
# image-stream MLP (ff)
W["ff.net.0.proj.weight"] = randn(MLP_HIDDEN, HIDDEN)
W["ff.net.0.proj.bias"] = randn(MLP_HIDDEN)
W["ff.net.2.weight"] = randn(HIDDEN, MLP_HIDDEN)
W["ff.net.2.bias"] = randn(HIDDEN)
# text-stream MLP (ff_context)
W["ff_context.net.0.proj.weight"] = randn(MLP_HIDDEN, HIDDEN)
W["ff_context.net.0.proj.bias"] = randn(MLP_HIDDEN)
W["ff_context.net.2.weight"] = randn(HIDDEN, MLP_HIDDEN)
W["ff_context.net.2.bias"] = randn(HIDDEN)


def heads_split(x):
    # x: (N, hidden) -> (HEADS, N, HEAD_DIM)
    n = x.shape[0]
    return x.reshape(n, HEADS, HEAD_DIM).transpose(1, 0, 2)


def heads_merge(x):
    # x: (HEADS, N, HEAD_DIM) -> (N, hidden)
    n = x.shape[1]
    return x.transpose(1, 0, 2).reshape(n, HIDDEN)


def joint_block_forward(img, txt, c):
    # ---- adaLN modulation for both streams from the SAME conditioning c ----
    mod_i = linear(silu(c), W["norm1.linear.weight"], W["norm1.linear.bias"])
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = np.split(mod_i, 6)
    mod_t = linear(silu(c), W["norm1_context.linear.weight"],
                   W["norm1_context.linear.bias"])
    c_sh_msa, c_sc_msa, c_g_msa, c_sh_mlp, c_sc_mlp, c_g_mlp = np.split(mod_t, 6)

    # ---- normed + modulated streams ----
    hi = modulate(layernorm_noaffine(img), sh_msa, sc_msa)
    ht = modulate(layernorm_noaffine(txt), c_sh_msa, c_sc_msa)

    # ---- per-stream q/k/v projections ----
    qi = linear(hi, W["attn.to_q.weight"], W["attn.to_q.bias"])
    ki = linear(hi, W["attn.to_k.weight"], W["attn.to_k.bias"])
    vi = linear(hi, W["attn.to_v.weight"], W["attn.to_v.bias"])
    qt = linear(ht, W["attn.add_q_proj.weight"], W["attn.add_q_proj.bias"])
    kt = linear(ht, W["attn.add_k_proj.weight"], W["attn.add_k_proj.bias"])
    vt = linear(ht, W["attn.add_v_proj.weight"], W["attn.add_v_proj.bias"])

    # ---- CONCATENATE along sequence axis: diffusers order is [img ; txt] ----
    q = np.concatenate([qi, qt], axis=0)   # (IMG_LEN+TXT_LEN, hidden)
    k = np.concatenate([ki, kt], axis=0)
    v = np.concatenate([vi, vt], axis=0)
    qh, kh, vh = heads_split(q), heads_split(k), heads_split(v)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    attn = softmax((qh @ kh.transpose(0, 2, 1)) * scale, axis=-1)
    o = heads_merge(attn @ vh)             # (IMG_LEN+TXT_LEN, hidden)

    # ---- split back per stream ----
    oi = o[:IMG_LEN]
    ot = o[IMG_LEN:]
    oi = linear(oi, W["attn.to_out.0.weight"], W["attn.to_out.0.bias"])
    ot = linear(ot, W["attn.to_add_out.weight"], W["attn.to_add_out.bias"])
    img = img + g_msa * oi
    txt = txt + c_g_msa * ot

    # ---- per-stream MLP ----
    hi = modulate(layernorm_noaffine(img), sh_mlp, sc_mlp)
    mi = linear(gelu_tanh(linear(hi, W["ff.net.0.proj.weight"],
                                 W["ff.net.0.proj.bias"])),
                W["ff.net.2.weight"], W["ff.net.2.bias"])
    img = img + g_mlp * mi
    ht = modulate(layernorm_noaffine(txt), c_sh_mlp, c_sc_mlp)
    mt = linear(gelu_tanh(linear(ht, W["ff_context.net.0.proj.weight"],
                                 W["ff_context.net.0.proj.bias"])),
                W["ff_context.net.2.weight"], W["ff_context.net.2.bias"])
    txt = txt + c_g_mlp * mt
    return img, txt


# ---------------- run a few (img, txt, c) cases ----------------
cases = []
for case in range(3):
    img = randn(IMG_LEN, HIDDEN, scale=1.0)
    txt = randn(TXT_LEN, HIDDEN, scale=1.0)
    c = randn(HIDDEN, scale=1.0)
    out_img, out_txt = joint_block_forward(img, txt, c)
    cases.append({
        "img": img.reshape(-1).tolist(),
        "txt": txt.reshape(-1).tolist(),
        "c": c.reshape(-1).tolist(),
        "out_img": out_img.reshape(-1).tolist(),
        "out_txt": out_txt.reshape(-1).tolist(),
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

tensors = {k: v.astype(np.float32) for k, v in W.items()}
save_file(tensors, os.path.join(fixtures, "tiny_mmdit.safetensors"))

config = {
    "_class_name": "SD3Transformer2DModel",
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
    "img_len": IMG_LEN,
    "txt_len": TXT_LEN,
    "mlp_hidden": MLP_HIDDEN,
    "layer_norm_eps": 1e-6,
}
with open(os.path.join(fixtures, "tiny_mmdit_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_mmdit_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_mmdit.safetensors,_config.json,_io.json to", fixtures)
print(f"  hidden={HIDDEN} heads={HEADS} head_dim={HEAD_DIM} "
      f"img_len={IMG_LEN} txt_len={TXT_LEN} mlp_hidden={MLP_HIDDEN}")
