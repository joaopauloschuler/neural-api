#!/usr/bin/env python3
"""Generate the Qwen-Image-2.1 parity oracle (tasklist A0) for the later Pascal
tasks A1-A7. Every model is a tiny RANDOM-weight version of the real component,
built from the real diffusers / transformers classes (never downloaded), with
the REAL tensor names and the real config layout.

Outputs (all under tests/fixtures/):

  tiny_qwenimage21/                   a pico diffusers folder (model_index.json
    scheduler/scheduler_config.json   copied from Qwen/Qwen-Image-2.1)
    text_encoder/{config.json,model.safetensors}      tiny qwen3_vl, BF16,
        config.json in the real (transformers 4.57) layout: rope_scaling with
        mrope_interleaved + mrope_section, rope_theta at the text level
    transformer/{config.json, 2 BF16 shards + .index.json}
    vae/{config.json,diffusion_pytorch_model.safetensors}   F32
  qwenimage21_scheduler_oracle.json        A1: sigmas, mu, timesteps, step()
  qwenimage21_prompt_tokens.json           A2: real template ids + drop_idx
                                           (only when the real tokenizer is
                                           reachable; skipped otherwise)
  tiny_qwenimage21_text_encoder_io.json    A2: last decoder layer before the
                                           final RMSNorm, M-RoPE and 1D RoPE
  tiny_qwenimage21_rope_io.json            A3: frame/h/w indices, complex table,
                                           rotated q
  tiny_qwenimage21_transformer_io.json     A4/A5: full, extract, cached, K/V,
                                           block 0 in cached mode, temb,
                                           modulation
  tiny_qwenimage21_vae_io.json             A6: latent de-normalisation + decode
  tiny_qwenimage21_vae_tiled_io.json       A6: 64x64 decode, whole and tiled
  tiny_qwenimage21_pipeline_io.json        A7: pico pipeline, fixed latents and
                                           fixed prompt embeds (32x64)
  tiny_qwenimage21_pipeline_64_io.json     A7: the same at 64x64

Precision. Weights are rounded to their storage dtype (BF16 for the text
encoder and transformer, F32 for the VAE) and every forward then runs in
float64. The reference code casts to float32 in a few places; this script
replaces exactly those casts with float64 versions of the same math (see
install_float64_shims) and records, per component, the max |difference| between
the float64 oracle and the unmodified reference ("native_hf_maxabs_diff").
The scheduler is left native (float32 sigmas, float32 sample in step()).

Tensors in the JSON files are {"shape": [...], "data": [flat row-major]}.
Complex tensors carry a trailing axis of 2 (real, imaginary).

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv, torch CPU build):
  ( ulimit -v 3145728; python3 tools/make_pico_qwenimage21_fixture.py \
      [--processor-dir DIR] )
DIR holds the real Qwen/Qwen-Image-2.1 `processor/` files; without it the
script downloads tokenizer.json, tokenizer_config.json and chat_template.jinja
(~11 MB) into the Hugging Face cache, or skips the token oracle when offline.
"""
import argparse
import copy
import json
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

import diffusers
from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                       QwenImage21Pipeline, QwenImage21Transformer2DModel)
from diffusers.models import normalization as diffusers_normalization
from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as vae_module
from diffusers.models.transformers import transformer_qwenimage21 as transformer_module
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
import transformers
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl_module

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")
PICO_DIR = os.path.join(FIXTURES, "tiny_qwenimage21")
REAL_REPO = "Qwen/Qwen-Image-2.1"

# Real scheduler/scheduler_config.json of Qwen/Qwen-Image-2.1.
SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.37.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
MODEL_INDEX = {
    "_class_name": "QwenImage21Pipeline",
    "_diffusers_version": "0.37.0.dev0",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                       "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")

# ---------------- pico sizes ----------------
TE_HIDDEN = 64
TE_LAYERS = 2
TE_HEADS = 4
TE_KV_HEADS = 2
TE_HEAD_DIM = 16
TE_FF = 96
TE_VOCAB = 300
TE_MROPE_SECTION = [4, 2, 2]          # sums to TE_HEAD_DIM / 2, like [24, 20, 20]
TE_TOKEN_IDS = [(37 * i + 11) % TE_VOCAB for i in range(14)]
TE_DROP_IDX = 5                       # stands in for the system-prompt length

TR_LAYERS = 2
TR_HEADS = 2
TR_HEAD_DIM = 16
TR_AXES = [4, 6, 6]                   # sums to TR_HEAD_DIM, like (16, 56, 56)
TR_CHANNELS = 16                      # transformer in/out channels == VAE z_dim
TR_GRID = (1, 4, 6)                   # target latent grid (frame, h, w)

VAE_Z = TR_CHANNELS
VAE_BASE = 2                          # real: encoder 96, decoder 144
VAE_DECODER_BASE = 3
VAE_DIM_MULT = [1, 2, 4, 8, 8]        # the real dim_mult: same shortcuts, 16x spatial
VAE_LATENT_HW = (2, 3)
VAE_LATENT_TILED_HW = (4, 4)          # a 64x64 image
VAE_TILINGS = [(32, 16), (48, 32)]    # (tile_sample_min, tile_sample_stride) in pixels
VAE_DUPUP_CASES = [(8, 8, 2), (8, 4, 2), (4, 2, 1), (3, 6, 2), (4, 4, 1)]  # (in, out, factor_t)

PIPE_HEIGHT, PIPE_WIDTH, PIPE_STEPS = 32, 64, 3
PIPE_64_SIZE = 64                     # the square dev-box size of the A7 example


# ---------------- JSON helpers ----------------
def tensor_json(x):
    x = torch.as_tensor(x).detach()
    if x.is_complex():
        x = torch.view_as_real(x.to(torch.complex128))
    if x.dtype == torch.bool or not x.is_floating_point():
        return {"shape": list(x.shape), "data": x.reshape(-1).to(torch.int64).tolist()}
    return {"shape": list(x.shape), "data": x.to(torch.float64).reshape(-1).tolist()}


def write_json(name, payload):
    path = os.path.join(FIXTURES, name)
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {name} ({os.path.getsize(path)} bytes)")


def max_abs_diff(a, b):
    return float((torch.as_tensor(a).to(torch.float64) - torch.as_tensor(b).to(torch.float64)).abs().max())


# ---------------- float64 versions of the reference's float32 casts ----------------
def diffusers_rmsnorm_f64(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    if self.weight is not None:
        hidden_states = hidden_states * self.weight
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
    return hidden_states


def zero_center_rmsnorm_f64(self, hidden_states):
    rrms = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
    return hidden_states * rrms * (self.weight + 1)


def temporal_timesteps_f64(self, timestep):
    half = self.timestep_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float64) / half)
    assert torch.allclose(freqs, self.freqs.to(torch.float64), rtol=1e-6), "max_period is not 10000"
    args = (self.time_factor * timestep.to(torch.float64))[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.timestep_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def rope_params_f64(self, index, dim, theta=10000):
    exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    freqs = torch.outer(index.to(torch.float64), 1.0 / torch.pow(float(theta), exponents))
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb_qwen_f64(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    assert not use_real, "Qwen-Image 2.1 calls the complex branch only"
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2).contiguous())
    x_out = torch.view_as_real(x_complex * freqs_cis.to(torch.complex128).unsqueeze(1)).flatten(3)
    return x_out.type_as(x)


def vae_upsample_f64(self, x):
    return nn.Upsample.forward(self, x)


def qwen3vl_rmsnorm_f64(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def qwen3vl_inv_freq_f64(config):
    base = config.rope_parameters["rope_theta"]
    dim = config.head_dim
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


def qwen3vl_mrope_f64(self, x, position_ids):
    inv_freq = qwen3vl_inv_freq_f64(self.config)
    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    freqs = (inv_freq_expanded @ position_ids[:, :, None, :].to(torch.float64)).transpose(2, 3)
    cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
    sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
    return cos.to(x.dtype), sin.to(x.dtype)


def qwen3vl_plain_rope_f64(self, x, position_ids):
    """1D RoPE over arange(seq_len): no M-RoPE section recomposition at all."""
    inv_freq = qwen3vl_inv_freq_f64(self.config)
    positions = torch.arange(position_ids.shape[-1], dtype=torch.float64)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)[None].expand(position_ids.shape[1], -1, -1)
    return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


SHIM_TARGETS = [
    (diffusers_normalization.RMSNorm, "forward", diffusers_rmsnorm_f64),
    (transformer_module.QwenImage21ZeroCenterRMSNorm, "forward", zero_center_rmsnorm_f64),
    (transformer_module.QwenImage21TemporalTimesteps, "forward", temporal_timesteps_f64),
    (transformer_module.QwenImage21Rope, "rope_params", rope_params_f64),
    (transformer_module, "apply_rotary_emb_qwen", apply_rotary_emb_qwen_f64),
    (vae_module.QwenImage21Upsample, "forward", vae_upsample_f64),
    (qwen3vl_module.Qwen3VLTextRMSNorm, "forward", qwen3vl_rmsnorm_f64),
    (qwen3vl_module.Qwen3VLTextRotaryEmbedding, "forward", qwen3vl_mrope_f64),
]
ORIGINALS = {(owner, name): getattr(owner, name) for owner, name, _ in SHIM_TARGETS}


def install_float64_shims(enabled=True):
    for owner, name, replacement in SHIM_TARGETS:
        setattr(owner, name, replacement if enabled else ORIGINALS[(owner, name)])


def rebuild_rope_tables(rope):
    """QwenImage21Rope builds its tables in __init__; rebuild them with the active rope_params."""
    pos_index = torch.arange(8192)
    neg_index = torch.arange(1024).flip(0) * -1 - 1
    rope.freqs = [torch.cat([rope.rope_params(pos_index, dim, rope.theta),
                             rope.rope_params(neg_index, dim, rope.theta)], dim=0)
                  for dim in rope.axes_dim]


class NativeReference:
    """Context: the unmodified reference code (float32 casts included)."""

    def __init__(self, *ropes):
        self.ropes = ropes

    def __enter__(self):
        install_float64_shims(False)
        for rope in self.ropes:
            rebuild_rope_tables(rope)

    def __exit__(self, *exc):
        install_float64_shims(True)
        for rope in self.ropes:
            rebuild_rope_tables(rope)


# ---------------- random weights ----------------
def randomize_parameters(module, seed, norm_suffixes=(), zero_centered_suffixes=(), out_scale=None):
    """fan-in scaled normal weights; norm gains 1 + 0.2 N(0,1); zero-centred gains 0.2 N(0,1)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            noise = torch.randn(param.shape, generator=gen, dtype=torch.float64)
            if name.endswith(zero_centered_suffixes):
                value = 0.2 * noise
            elif name.endswith(norm_suffixes):
                value = 1.0 + 0.2 * noise
            elif param.dim() >= 2:
                value = noise / math.sqrt(param[0].numel())
            else:
                value = 0.1 * noise
            if out_scale is not None and name in out_scale:
                value = value * out_scale[name]
            param.copy_(value.to(param.dtype))


def round_parameters(module, storage_dtype):
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(param.to(storage_dtype).to(param.dtype))


# ---------------- A1: scheduler ----------------
def scheduler_float64(num_steps, mu, sample, velocities):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
    one_minus = 1.0 - shifted
    shifted = 1.0 - one_minus / (one_minus[-1] / (1.0 - SCHEDULER_CONFIG["shift_terminal"]))
    sigmas_full = np.append(shifted, 0.0)
    steps = []
    x = sample.copy()
    for index, velocity in enumerate(velocities):
        x = x + (sigmas_full[index + 1] - sigmas_full[index]) * velocity
        steps.append(x.copy())
    return shifted * 1000.0, sigmas_full, steps


def make_scheduler_oracle():
    gen = torch.Generator().manual_seed(20260923)
    cases = []
    for num_steps, image_seq_len in [(40, 4096), (4, 24), (10, 256), (20, 8192), (8, 16384), (3, 8)]:
        mu = calculate_shift(image_seq_len, SCHEDULER_CONFIG["base_image_seq_len"],
                             SCHEDULER_CONFIG["max_image_seq_len"], SCHEDULER_CONFIG["base_shift"],
                             SCHEDULER_CONFIG["max_shift"])
        sigmas_in = np.linspace(1.0, 1 / num_steps, num_steps)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
        scheduler.set_timesteps(sigmas=sigmas_in, mu=mu)
        scheduler.set_begin_index(0)
        sample = torch.randn(6, generator=gen, dtype=torch.float64)
        velocities = [torch.randn(6, generator=gen, dtype=torch.float64) for _ in range(2)]
        hf_steps = []
        x = sample
        for index, velocity in enumerate(velocities):
            x = scheduler.step(velocity, scheduler.timesteps[index], x, return_dict=False)[0]
            hf_steps.append(x)
        timesteps64, sigmas64, steps64 = scheduler_float64(
            num_steps, mu, sample.numpy(), [v.numpy() for v in velocities])
        assert max_abs_diff(scheduler.sigmas, sigmas64) < 1e-6
        assert max_abs_diff(hf_steps[-1], steps64[-1]) < 1e-5
        cases.append({
            "num_steps": num_steps,
            "image_seq_len": image_seq_len,
            "mu": mu,
            "sigmas_linspace": sigmas_in.tolist(),
            "timesteps": scheduler.timesteps.to(torch.float64).tolist(),
            "sigmas": scheduler.sigmas.to(torch.float64).tolist(),
            "timesteps_f64": timesteps64.tolist(),
            "sigmas_f64": sigmas64.tolist(),
            "step_sample": sample.tolist(),
            "step_velocity": [v.tolist() for v in velocities],
            "step_out": [s.tolist() for s in hf_steps],
            "step_out_f64": [s.tolist() for s in steps64],
        })
    write_json("qwenimage21_scheduler_oracle.json", {
        "config": SCHEDULER_CONFIG,
        "note": "timesteps/sigmas/step_out: diffusers as run (float32 sigmas, step() casts the sample "
                "to float32); *_f64: the same formulas in float64. step_out[i] is the sample after "
                "step i starting from step_sample with begin index 0.",
        "cases": cases,
    })


# ---------------- A2: text encoder ----------------
def text_encoder_config():
    """The real text_encoder/config.json layout with pico sizes."""
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_id": 290,
        "model_type": "qwen3_vl",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 297,
            "dtype": "bfloat16",
            "eos_token_id": 298,
            "head_dim": TE_HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": TE_HIDDEN,
            "initializer_range": 0.02,
            "intermediate_size": TE_FF,
            "max_position_embeddings": 4096,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": TE_HEADS,
            "num_hidden_layers": TE_LAYERS,
            "num_key_value_heads": TE_KV_HEADS,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": TE_MROPE_SECTION,
                             "rope_type": "default"},
            "rope_theta": 5000000,
            "use_cache": True,
            "vocab_size": TE_VOCAB,
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.57.1",
        "video_token_id": 291,
        "vision_config": {
            "deepstack_visual_indexes": [0],
            "depth": 1,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "model_type": "qwen3_vl",
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": TE_HIDDEN,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 293,
        "vision_start_token_id": 292,
    }


def last_layer_before_norm(model, input_ids):
    """The pipeline's hook trick: hidden_states[-1] with the final RMSNorm made an identity."""
    text_model = model.model.language_model
    layer_outputs = []
    handles = [text_model.norm.register_forward_hook(lambda module, args, output: args[0]),
               text_model.layers[-1].register_forward_hook(
                   lambda module, args, output: layer_outputs.append(output))]
    try:
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    last_layer = layer_outputs[0][0] if isinstance(layer_outputs[0], tuple) else layer_outputs[0]
    assert max_abs_diff(outputs.hidden_states[-1], last_layer) == 0.0
    return outputs.hidden_states[-1]


def make_text_encoder():
    config_dict = text_encoder_config()
    torch.manual_seed(1)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config_dict)).to(torch.float64).eval()
    randomize_parameters(model, 20260923, norm_suffixes=("layernorm.weight", "norm.weight"))
    round_parameters(model, torch.bfloat16)

    out_dir = os.path.join(PICO_DIR, "text_encoder")
    copy.deepcopy(model).to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    for extra in os.listdir(out_dir):
        if extra not in ("model.safetensors",):
            os.remove(os.path.join(out_dir, extra))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    names = sorted(load_file(os.path.join(out_dir, "model.safetensors")).keys())
    assert any(n.startswith("model.language_model.layers.") for n in names), names[:5]
    assert any(n.startswith("model.visual.") for n in names) and "lm_head.weight" in names

    # The committed files must reproduce the in-memory model.
    reloaded = Qwen3VLForConditionalGeneration.from_pretrained(out_dir, dtype=torch.float64).eval()
    input_ids = torch.tensor([TE_TOKEN_IDS])
    with torch.no_grad():
        hidden_mrope = last_layer_before_norm(model, input_ids)
        hidden_reloaded = last_layer_before_norm(reloaded, input_ids)
        final_norm = model.model.language_model.norm(hidden_mrope)
        qwen3vl_module.Qwen3VLTextRotaryEmbedding.forward = qwen3vl_plain_rope_f64
        hidden_plain = last_layer_before_norm(model, input_ids)
        install_float64_shims(True)
        with NativeReference():
            hidden_native = last_layer_before_norm(model, input_ids)
    assert max_abs_diff(hidden_mrope, hidden_reloaded) == 0.0
    mrope_vs_plain = max_abs_diff(hidden_mrope, hidden_plain)
    assert mrope_vs_plain < 1e-12, mrope_vs_plain
    rope_section_check = model.config.text_config.rope_parameters
    assert rope_section_check["mrope_interleaved"] and rope_section_check["mrope_section"] == TE_MROPE_SECTION

    embeds = hidden_mrope[0, TE_DROP_IDX:]
    write_json("tiny_qwenimage21_text_encoder_io.json", {
        "token_ids": TE_TOKEN_IDS,
        "drop_idx": TE_DROP_IDX,
        "hidden_last_layer": tensor_json(hidden_mrope[0]),
        "hidden_last_layer_1d_rope": tensor_json(hidden_plain[0]),
        "hidden_after_final_norm": tensor_json(final_norm[0]),
        "prompt_embeds": tensor_json(embeds),
        "mrope_vs_1d_rope_maxabs_diff": mrope_vs_plain,
        "native_hf_maxabs_diff": max_abs_diff(hidden_mrope, hidden_native),
        "note": "hidden_last_layer = output of the last decoder layer before the final RMSNorm, "
                "positions 0..L-1 (text-only M-RoPE); hidden_last_layer_1d_rope = same weights with "
                "plain rotate-half 1D RoPE; prompt_embeds = hidden_last_layer[drop_idx:].",
    })
    return model, embeds


def make_prompt_token_oracle(processor_dir):
    from transformers import AutoTokenizer
    try:
        if processor_dir is None:
            from huggingface_hub import hf_hub_download
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                path = hf_hub_download(REAL_REPO, f"processor/{name}")
            processor_dir = os.path.dirname(path)
        tokenizer = AutoTokenizer.from_pretrained(processor_dir)
        with open(os.path.join(processor_dir, "chat_template.jinja")) as f:
            chat_template = f.read()
    except Exception as error:  # offline or gated: the oracle is optional
        print(f"SKIPPED qwenimage21_prompt_tokens.json: {error}")
        return None
    sys_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    sys_tokens = tokenizer.apply_chat_template(sys_message, chat_template=chat_template, tokenize=True,
                                               return_dict=False)
    if isinstance(sys_tokens, dict) or hasattr(sys_tokens, "input_ids"):
        sys_tokens = sys_tokens["input_ids"]
    if sys_tokens and isinstance(sys_tokens[0], list):
        sys_tokens = sys_tokens[0]
    cases = []
    for prompt in ["A red fox in the snow", "A capybara wearing a wizard hat, reading a book by "
                   "candlelight, oil painting", " "]:
        text = PROMPT_TEMPLATE_T2I.format(prompt)
        ids = tokenizer(text)["input_ids"]
        cases.append({"prompt": prompt, "template_text": text, "input_ids": ids,
                      "embed_ids": ids[len(sys_tokens):]})
    write_json("qwenimage21_prompt_tokens.json", {
        "system_prompt": SYSTEM_PROMPT,
        "system_chat_template_text": tokenizer.apply_chat_template(sys_message, chat_template=chat_template,
                                                                   tokenize=False),
        "system_token_ids": sys_tokens,
        "drop_idx": len(sys_tokens),
        "image_pad_token_id": tokenizer.encode("<|image_pad|>")[0],
        "cases": cases,
        "note": "Tokenized with the real Qwen/Qwen-Image-2.1 processor/tokenizer.json via AutoTokenizer "
                "(Qwen3VLProcessor needs torchvision for its video processor; text-only prompts do not "
                "reach it). drop_idx = len(apply_chat_template(system message)), as the pipeline computes it.",
    })
    return len(sys_tokens)


# ---------------- A3: RoPE ----------------
def rope_index_table(img_shapes, image_pad_mask):
    """QwenImage21Rope.forward with tables that hold their own row index: returns (seq, 3) positions."""
    index_rope = transformer_module.QwenImage21Rope(theta=10000, axes_dim=[2, 2, 2])
    index_column = torch.cat([torch.arange(8192), torch.arange(1024).flip(0) * -1 - 1]).to(torch.float64)
    index_rope.freqs = [index_column[:, None]] * 3
    return index_rope(img_shapes, image_pad_mask, device="cpu").to(torch.int64)


def make_rope_oracle():
    rope = transformer_module.QwenImage21Rope(theta=10000, axes_dim=TR_AXES)
    gen = torch.Generator().manual_seed(7)
    cases = []
    for label, text_before, grid, text_after in [("odd_grid", 3, (1, 3, 5), 0),
                                                  ("even_grid", 3, (1, 4, 2), 0),
                                                  ("trailing_text", 2, (1, 2, 3), 2)]:
        mask = torch.tensor([False] * text_before + [True] * (grid[1] * grid[2]) + [False] * text_after)
        table = rope([grid], mask, device="cpu")
        query = torch.randn(1, mask.numel(), TR_HEADS, TR_HEAD_DIM, generator=gen, dtype=torch.float64)
        rotated = transformer_module.apply_rotary_emb_qwen(query, table, use_real=False)
        with NativeReference(rope):
            native = ORIGINALS[(transformer_module, "apply_rotary_emb_qwen")](
                query, rope([grid], mask, device="cpu"), use_real=False)
        cases.append({
            "label": label,
            "img_shapes": [list(grid)],
            "image_pad_mask": mask.to(torch.int64).tolist(),
            "positions_fhw": tensor_json(rope_index_table([grid], mask)),
            "freqs_cis": tensor_json(table),
            "query": tensor_json(query),
            "query_rotated": tensor_json(rotated),
            "native_hf_maxabs_diff": max_abs_diff(rotated, native),
        })
    write_json("tiny_qwenimage21_rope_io.json", {
        "theta": 10000,
        "axes_dim": TR_AXES,
        "note": "positions_fhw[s] = (frame, h, w) index of token s; freqs_cis[s] = complex table "
                "cat(frame axis, h axis, w axis) with axes_dim/2 entries each; query is (B, S, heads, "
                "head_dim), rotated in consecutive (real, imag) pairs by apply_rotary_emb_qwen(use_real=False).",
        "cases": cases,
    })


# ---------------- A4/A5: transformer ----------------
def target_img_mask(text_len, latent_tokens):
    """img_mask exactly as the text-to-image pipeline builds it: no image slots in the prompt,
    then latent_tokens // 4 target slots appended."""
    return torch.cat([torch.zeros(1, text_len, dtype=torch.bool),
                      torch.ones(1, latent_tokens // 4, dtype=torch.bool)], dim=1)


def make_transformer(context_in_dim):
    torch.manual_seed(2)
    model = QwenImage21Transformer2DModel(
        patch_size=1, in_channels=TR_CHANNELS, out_channels=TR_CHANNELS, num_layers=TR_LAYERS,
        attention_head_dim=TR_HEAD_DIM, num_attention_heads=TR_HEADS, context_in_dim=context_in_dim,
        mlp_ratio=3, axes_dims_rope=tuple(TR_AXES), eps=1e-6, causal_condition=True)
    model = model.to(torch.float64).eval()
    randomize_parameters(model, 20260924, norm_suffixes=("norm_q.weight", "norm_k.weight"),
                         zero_centered_suffixes=("text_norm.weight",))
    round_parameters(model, torch.bfloat16)

    out_dir = os.path.join(PICO_DIR, "transformer")
    total_bytes = sum(p.numel() * 2 for p in model.parameters())
    copy.deepcopy(model).to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True,
                                                           max_shard_size=total_bytes * 6 // 10)
    shards = sorted(n for n in os.listdir(out_dir) if n.endswith(".safetensors"))
    assert len(shards) == 2, shards
    reloaded = QwenImage21Transformer2DModel.from_pretrained(out_dir, torch_dtype=torch.float64).eval()
    rebuild_rope_tables(reloaded.pos_embed)
    return model, reloaded


def run_transformer(model, latents, embeds, t, kv_cache=None, kv_cache_mode=None):
    img_mask = target_img_mask(embeds.shape[1], latents.shape[1])
    return model(hidden_states=latents, encoder_hidden_states=embeds, timestep=t,
                 img_shapes=[[TR_GRID]], img_mask=img_mask, kv_cache=kv_cache,
                 kv_cache_mode=kv_cache_mode, return_dict=False)[0]


def make_transformer_oracle(model, reloaded, embeds):
    tokens = TR_GRID[1] * TR_GRID[2]
    gen = torch.Generator().manual_seed(11)
    latents_1 = torch.randn(1, tokens, TR_CHANNELS, generator=gen, dtype=torch.float64)
    latents_2 = torch.randn(1, tokens, TR_CHANNELS, generator=gen, dtype=torch.float64)
    t_1 = torch.tensor([0.9], dtype=torch.float64)
    t_2 = torch.tensor([0.35], dtype=torch.float64)

    captured = {}

    def capture(key):
        def hook(module, args, output):
            captured.setdefault(key, []).append(output.detach().clone())
        return hook

    def capture_block(module, args, kwargs, output):
        captured["block0"] = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        captured["block0"]["output"] = output.detach().clone()

    handles = [model.time_text_embed.register_forward_hook(capture("temb")),
               model.modulation.register_forward_hook(capture("modulation"))]
    with torch.no_grad():
        full = run_transformer(model, latents_1, embeds, t_1)
        full_reloaded = run_transformer(reloaded, latents_1, embeds, t_1)
        cache = QwenImage21KVCache(TR_LAYERS)
        extract = run_transformer(model, latents_1, embeds, t_1, cache, "extract")
        block_handle = model.transformer_blocks[0].register_forward_hook(capture_block, with_kwargs=True)
        cached = run_transformer(model, latents_2, embeds, t_2, cache, "cached")
        block_handle.remove()
        full_2 = run_transformer(model, latents_2, embeds, t_2)
        with NativeReference(model.pos_embed):
            native = run_transformer(model, latents_1, embeds, t_1)
    for handle in handles:
        handle.remove()

    prefix_len = embeds.shape[1]
    assert max_abs_diff(full, full_reloaded) == 0.0
    assert max_abs_diff(full, extract) < 1e-12
    cached_vs_full = max_abs_diff(cached, full_2[:, prefix_len:])
    assert cached_vs_full < 1e-12, cached_vs_full

    block = captured["block0"]
    k0, v0 = cache.get_layer(0).get()
    write_json("tiny_qwenimage21_transformer_io.json", {
        "img_shapes": [list(TR_GRID)],
        "text_len": prefix_len,
        "img_mask": target_img_mask(prefix_len, tokens).to(torch.int64)[0].tolist(),
        "encoder_hidden_states": tensor_json(embeds[0]),
        "timestep_1": float(t_1), "timestep_2": float(t_2),
        "latents_1": tensor_json(latents_1[0]),
        "latents_2": tensor_json(latents_2[0]),
        "full_output": tensor_json(full[0]),
        "extract_output": tensor_json(extract[0]),
        "cached_output": tensor_json(cached[0]),
        "cache_k": [tensor_json(cache.get_layer(i).k[0]) for i in range(TR_LAYERS)],
        "cache_v": [tensor_json(cache.get_layer(i).v[0]) for i in range(TR_LAYERS)],
        "temb_1": tensor_json(captured["temb"][0]),
        "modulation_1": tensor_json(captured["modulation"][0]),
        "temb_2": tensor_json(captured["temb"][2]),
        "modulation_2": tensor_json(captured["modulation"][2]),
        "block0_cached": {
            "hidden_in": tensor_json(block["hidden_states"][0]),
            "modulation": tensor_json(block["modulation"]),
            "rotary": tensor_json(block["rotary_emb"]),
            "target_token_mask": tensor_json(block["target_token_mask"]),
            "cache_k": tensor_json(k0[0]),
            "cache_v": tensor_json(v0[0]),
            "output": tensor_json(block["output"][0]),
        },
        "cached_vs_full_maxabs_diff": cached_vs_full,
        "native_hf_maxabs_diff": max_abs_diff(full, native),
        "note": "full_output = forward without cache over the joint sequence (text rows first, then "
                "target rows); extract_output = same call in extract mode (fills cache_k/cache_v, "
                "(text_len, heads, head_dim) post-norm post-RoPE); cached_output = target rows only, "
                "second step with timestep_2 and latents_2. temb/modulation rows: [sampled t, t=0]. "
                "modulation = [scale1 | gate1 | scale2 | gate2]. timestep is t in [0,1] (pipeline t/1000).",
    })


# ---------------- A6: VAE ----------------
def make_vae():
    gen = torch.Generator().manual_seed(5)
    latents_mean = (torch.randn(VAE_Z, generator=gen, dtype=torch.float64) * 0.5).tolist()
    latents_std = (1.0 + torch.rand(VAE_Z, generator=gen, dtype=torch.float64) * 3.0).tolist()
    latents_mean = [float(np.float32(v)) for v in latents_mean]
    latents_std = [float(np.float32(v)) for v in latents_std]
    torch.manual_seed(3)
    vae = AutoencoderKLQwenImage21(
        base_dim=VAE_BASE, decoder_base_dim=VAE_DECODER_BASE, z_dim=VAE_Z, dim_mult=VAE_DIM_MULT,
        num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True, True], dropout=0.0,
        latents_mean=latents_mean, latents_std=latents_std, is_residual=True, in_channels=4,
        out_channels=4, patch_size=None, scale_factor_temporal=8, scale_factor_spatial=16)
    vae = vae.to(torch.float64).eval()
    randomize_parameters(vae, 20260925, norm_suffixes=("gamma",),
                         out_scale={"decoder.conv_out.weight": 0.25, "decoder.conv_out.bias": 0.25})
    round_parameters(vae, torch.float32)
    out_dir = os.path.join(PICO_DIR, "vae")
    copy.deepcopy(vae).to(torch.float32).save_pretrained(out_dir, safe_serialization=True)
    reloaded = AutoencoderKLQwenImage21.from_pretrained(out_dir, torch_dtype=torch.float64).eval()
    return vae, reloaded


def count_time_conv_calls(vae):
    calls = []
    handles = [module.time_conv.register_forward_hook(lambda m, a, o, name=name: calls.append(name))
               for name, module in vae.named_modules()
               if isinstance(module, vae_module.QwenImage21Resample) and hasattr(module, "time_conv")]
    return calls, handles


def denormalize_latents(vae, latents):
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, VAE_Z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, VAE_Z, 1, 1, 1)
    return latents * std + mean


def make_vae_oracle(vae, reloaded):
    gen = torch.Generator().manual_seed(13)
    latents = torch.randn(1, VAE_Z, 1, *VAE_LATENT_HW, generator=gen, dtype=torch.float64)
    denormalized = denormalize_latents(vae, latents)
    raw = []
    raw_handle = vae.decoder.register_forward_hook(lambda m, a, o: raw.append(o.detach().clone()))
    calls, handles = count_time_conv_calls(vae)
    with torch.no_grad():
        decoded = vae.decode(denormalized, return_dict=False)[0]
        decoded_reloaded = reloaded.decode(denormalized, return_dict=False)[0]
        with NativeReference():
            native = vae.decode(denormalized, return_dict=False)[0]
    raw_handle.remove()
    for handle in handles:
        handle.remove()
    time_conv_modules = sorted({name for name, module in vae.named_modules()
                                if isinstance(module, vae_module.QwenImage21Resample)
                                and hasattr(module, "time_conv")})
    assert max_abs_diff(decoded, decoded_reloaded) == 0.0
    clamped_fraction = float((raw[0].abs() >= 1.0).to(torch.float64).mean())
    upsample_modes = [(name, module.mode) for name, module in vae.decoder.named_modules()
                      if isinstance(module, vae_module.QwenImage21Resample)]
    write_json("tiny_qwenimage21_vae_io.json", {
        "latents_normalized": tensor_json(latents[0, :, 0]),
        "latents_denormalized": tensor_json(denormalized[0, :, 0]),
        "decoder_raw": tensor_json(raw[0][0, :, 0]),
        "decoded": tensor_json(decoded[0, :, 0]),
        "time_conv_calls_during_decode": len(calls),
        "time_conv_modules": [name for name in time_conv_modules],
        "decoder_resample_modes": [list(item) for item in upsample_modes],
        "clamped_fraction": clamped_fraction,
        "native_hf_maxabs_diff": max_abs_diff(decoded, native),
        "note": "latents_denormalized = latents_normalized * latents_std + latents_mean (per channel, "
                "vae/config.json). decoder_raw = decoder output before the [-1,1] clamp (RGBA, (C,H,W)); "
                "decoded = vae.decode output after the clamp. The single frame axis is dropped.",
    })
    return len(calls)


def make_vae_tiled_oracle(vae):
    """A 4x4 latent (64x64 image) decoded whole and with diffusers' tiled_decode at two tilings."""
    gen = torch.Generator().manual_seed(19)
    latents = torch.randn(1, VAE_Z, 1, *VAE_LATENT_TILED_HW, generator=gen, dtype=torch.float64)
    denormalized = denormalize_latents(vae, latents)
    tiled_vae = copy.deepcopy(vae)
    cases = []
    with torch.no_grad():
        whole = vae.decode(denormalized, return_dict=False)[0]
        for tile_size, tile_stride in VAE_TILINGS:
            tiled_vae.enable_tiling(tile_sample_min_height=tile_size, tile_sample_min_width=tile_size,
                                    tile_sample_stride_height=tile_stride,
                                    tile_sample_stride_width=tile_stride)
            tiled = tiled_vae.decode(denormalized, return_dict=False)[0]
            with NativeReference():
                native = tiled_vae.decode(denormalized, return_dict=False)[0]
            cases.append({"tile_sample_size": tile_size, "tile_sample_stride": tile_stride,
                          "decoded": tensor_json(tiled[0, :, 0]),
                          "tiled_vs_whole_maxabs_diff": max_abs_diff(tiled, whole),
                          "native_hf_maxabs_diff": max_abs_diff(tiled, native)})
    dupup_cases = []
    for in_channels, out_channels, factor_t in VAE_DUPUP_CASES:
        dupup_input = torch.randn(1, in_channels, 1, 2, 3, generator=gen, dtype=torch.float64)
        dupup = vae_module.QwenImage21DupUp3D(in_channels, out_channels, factor_t=factor_t, factor_s=2)
        dupup_output = dupup(dupup_input, first_chunk=True)
        dupup_cases.append({"in_channels": in_channels, "out_channels": out_channels, "factor_t": factor_t,
                            "input": tensor_json(dupup_input[0, :, 0]),
                            "output": tensor_json(dupup_output[0, :, 0])})
    write_json("tiny_qwenimage21_vae_tiled_io.json", {
        "latents_normalized": tensor_json(latents[0, :, 0]),
        "decoded_whole": tensor_json(whole[0, :, 0]),
        "tilings": cases,
        "dupup_cases": dupup_cases,
        "note": "decoded_whole = vae.decode(latents * latents_std + latents_mean) without tiling; "
                "tilings[i].decoded = the same with vae.enable_tiling(tile_sample_min = tile_sample_size, "
                "tile_sample_stride = tile_sample_stride) on both axes. RGBA (C,H,W) after the [-1,1] clamp. "
                "dupup_cases: QwenImage21DupUp3D(in, out, factor_t, factor_s=2)(input, first_chunk=True) on "
                "one frame, (C,H,W).",
    })


# ---------------- A7: pico pipeline ----------------
class TemplateOnlyProcessor:
    """QwenImage21Pipeline.__init__ reads drop_idx and the <|image_pad|> id from its processor; with
    prompt_embeds supplied nothing else touches it, so the pico pipeline needs only these two answers."""

    class _Tokenizer:
        def encode(self, text):
            return [290]

    def __init__(self, drop_idx):
        self.drop_idx = drop_idx
        self.tokenizer = self._Tokenizer()

    def apply_chat_template(self, messages, tokenize=True, return_dict=False):
        return [[0] * self.drop_idx]


def make_pipeline_oracle(transformer, vae, text_encoder, embeds, height=PIPE_HEIGHT, width=PIPE_WIDTH,
                         file_name="tiny_qwenimage21_pipeline_io.json"):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                               processor=TemplateOnlyProcessor(TE_DROP_IDX), transformer=transformer)
    latent_h, latent_w = 2 * (height // 32), 2 * (width // 32)
    gen = torch.Generator().manual_seed(17)
    initial = torch.randn(1, latent_h * latent_w, TR_CHANNELS, generator=gen, dtype=torch.float64)
    prompt_embeds = embeds[None].clone()

    def run(use_kv_cache):
        step_latents, timesteps, decode_in, decode_out = [], [], [], []

        def on_step(pipeline, index, t, callback_kwargs):
            step_latents.append(callback_kwargs["latents"].clone())
            timesteps.append(float(t))
            return callback_kwargs

        original_decode = vae.decode

        def recording_decode(z, return_dict=True):
            decode_in.append(z.clone())
            result = original_decode(z, return_dict=return_dict)
            decode_out.append((result[0] if not return_dict else result.sample).clone())
            return result

        vae.decode = recording_decode
        try:
            image = pipe(prompt_embeds=prompt_embeds, height=height, width=width,
                         num_inference_steps=PIPE_STEPS, latents=initial.clone(), output_type="pt",
                         callback_on_step_end=on_step, use_kv_cache=use_kv_cache).images
        finally:
            del vae.decode
        return image, step_latents, timesteps, decode_in[0], decode_out[0], list(pipe.scheduler.sigmas)

    with torch.no_grad():
        image, step_latents, timesteps, decode_in, decode_out, sigmas = run(True)
        image_nocache, step_latents_nocache, *_ = run(False)
    cache_vs_nocache = max_abs_diff(step_latents[-1], step_latents_nocache[-1])
    assert cache_vs_nocache < 1e-12, cache_vs_nocache
    write_json(file_name, {
        "height": height, "width": width, "num_inference_steps": PIPE_STEPS,
        "latent_grid": [1, latent_h, latent_w],
        "prompt_embeds": tensor_json(prompt_embeds[0]),
        "initial_latents": tensor_json(initial[0]),
        "timesteps": timesteps,
        "sigmas": [float(s) for s in sigmas],
        "step_latents": [tensor_json(latents[0]) for latents in step_latents],
        "final_latents": tensor_json(step_latents[-1][0]),
        "vae_input_denormalized": tensor_json(decode_in[0, :, 0]),
        "decoded": tensor_json(decode_out[0, :, 0]),
        "image": tensor_json(image[0]),
        "use_kv_cache_vs_off_maxabs_diff": cache_vs_nocache,
        "note": "true_cfg_scale 1 (one transformer pass per step), use_kv_cache on (extract at step 0, "
                "cached after). Latents are packed (tokens, channels), token = h * latent_w + w. "
                "step_latents[i] = latents after scheduler step i (native scheduler: float32 sigmas, "
                "float32 sample inside step()). decoded = vae.decode output in [-1,1] (RGBA, (C,H,W)); "
                "image = pipeline output_type='pt', i.e. decoded / 2 + 0.5 clamped to [0,1].",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processor-dir", default=None,
                        help="local copy of Qwen/Qwen-Image-2.1 processor/ for the real token oracle")
    args = parser.parse_args()
    torch.set_num_threads(1)
    install_float64_shims(True)
    if os.path.isdir(PICO_DIR):
        shutil.rmtree(PICO_DIR)
    os.makedirs(os.path.join(PICO_DIR, "scheduler"))
    with open(os.path.join(PICO_DIR, "model_index.json"), "w") as f:
        json.dump(MODEL_INDEX, f, indent=2)
    with open(os.path.join(PICO_DIR, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(SCHEDULER_CONFIG, f, indent=2)
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
          f"diffusers {diffusers.__version__}")

    make_scheduler_oracle()
    make_prompt_token_oracle(args.processor_dir)
    text_encoder, embeds = make_text_encoder()
    make_rope_oracle()
    transformer, transformer_reloaded = make_transformer(context_in_dim=TE_HIDDEN)
    make_transformer_oracle(transformer, transformer_reloaded, embeds[None])
    vae, vae_reloaded = make_vae()
    time_conv_calls = make_vae_oracle(vae, vae_reloaded)
    make_pipeline_oracle(transformer, vae, text_encoder, embeds)
    make_vae_tiled_oracle(vae)
    # Last, so every fixture above is unchanged by its addition.
    make_pipeline_oracle(transformer, vae, text_encoder, embeds, PIPE_64_SIZE, PIPE_64_SIZE,
                         "tiny_qwenimage21_pipeline_64_io.json")
    print(f"time_conv calls during a single-image decode: {time_conv_calls}")


if __name__ == "__main__":
    main()
