#!/usr/bin/env python3
"""Tiny DeepSeek-V2 YaRN-with-mscale fixture generator (HF float64 oracle).

Coded by Claude (AI).

Run with the shared venv (torch + transformers, CPU):
    /home/bpsa/x/bin/python tools/deepseek_v2_yarn_tiny_fixture.py

Sibling of tools/deepseek_v2_tiny_fixture.py. Identical pico DeepSeek-V2
architecture, but the config carries an EXPLICIT YaRN rope_scaling with
DeepSeek-style mscale / mscale_all_dim attention-scale overrides and a
factor > 1. HF transformers' _compute_yarn_parameters folds these into the
single post-rotation cos/sin "attention_factor":
    _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(factor, mscale_all_dim)
where _yarn_get_mscale(s, m) = 1 if s <= 1 else 0.1*m*ln(s)+1. The chosen
mscale=2.0 / mscale_all_dim=0.5 give 1.1944... which is DELIBERATELY distinct
from the YaRN default 0.1*ln(factor)+1 = 1.1386..., so the parity test would
FAIL if neuralpretrained.pas ignored the override (the regression this guards).

Generates tests/fixtures/tiny_deepseek_v2_yarn.{safetensors,config.json,
logits.json}. Same re-randomization / routing-gap / non-vacuity guards as the
base fixture.
"""
import json
import math
import os

import torch
from safetensors.torch import save_file
from transformers.models.deepseek_v2 import (DeepseekV2Config,
                                             DeepseekV2ForCausalLM)

SEED = 99999
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "tests", "fixtures")
BASE = "tiny_deepseek_v2_yarn"

N_LAYER = 2
N_HEAD = 2
HIDDEN = 8
QK_NOPE = 6
QK_ROPE = 4
V_HEAD = 6
KV_LORA_RANK = 5
INTERMEDIATE = 12
MOE_INTERMEDIATE = 5
N_SHARED = 2
N_ROUTED = 4
TOP_K = 2
FIRST_K_DENSE = 1
ROUTED_SCALE = 1.5
MAX_POS = 16
VOCAB = 11
ROPE_THETA = 10000.0
RMS_EPS = 1e-5
N_SEQ = 3

# YaRN with explicit mscale/mscale_all_dim. factor>1 and an
# original_max_position_embeddings < MAX_POS so the band ramp is non-trivial.
YARN_FACTOR = 4.0
YARN_ORIG_MAXPOS = 4
MSCALE = 2.0
MSCALE_ALL_DIM = 0.5


def yarn_get_mscale(scale, m):
    if scale <= 1:
        return 1.0
    return 0.1 * m * math.log(scale) + 1.0


CONFIG_JSON = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "num_hidden_layers": N_LAYER,
    "num_attention_heads": N_HEAD,
    "num_key_value_heads": N_HEAD,
    "qk_nope_head_dim": QK_NOPE,
    "qk_rope_head_dim": QK_ROPE,
    "v_head_dim": V_HEAD,
    "kv_lora_rank": KV_LORA_RANK,
    "q_lora_rank": None,
    "intermediate_size": INTERMEDIATE,
    "moe_intermediate_size": MOE_INTERMEDIATE,
    "n_shared_experts": N_SHARED,
    "n_routed_experts": N_ROUTED,
    "num_experts_per_tok": TOP_K,
    "first_k_dense_replace": FIRST_K_DENSE,
    "moe_layer_freq": 1,
    "norm_topk_prob": False,
    "routed_scaling_factor": ROUTED_SCALE,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    "max_position_embeddings": MAX_POS,
    "rope_theta": ROPE_THETA,
    "rope_scaling": {
        "type": "yarn",
        "factor": YARN_FACTOR,
        "original_max_position_embeddings": YARN_ORIG_MAXPOS,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": MSCALE,
        "mscale_all_dim": MSCALE_ALL_DIM,
    },
    "rms_norm_eps": RMS_EPS,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def build_model():
    cfg = DeepseekV2Config(**{k: v for k, v in CONFIG_JSON.items()
                              if k not in ("architectures", "torch_dtype")})
    torch.manual_seed(SEED)
    model = DeepseekV2ForCausalLM(cfg)
    model.config._attn_implementation = "eager"
    model.config._experts_implementation = "eager"
    gen = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for name, par in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                par.copy_(1.0 + 0.1 * torch.randn(par.shape, generator=gen))
            elif "embed_tokens" in name or name == "lm_head.weight":
                par.copy_(0.4 * torch.randn(par.shape, generator=gen))
            else:
                par.copy_(0.3 * torch.randn(par.shape, generator=gen))
    model.double().eval()
    return model


def hub_state_dict(model):
    out = {}
    for key, val in model.state_dict().items():
        if key.endswith("mlp.experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            for e in range(val.shape[0]):
                gate, up = val[e].chunk(2, dim=0)
                out[f"{prefix}{e}.gate_proj.weight"] = gate.contiguous()
                out[f"{prefix}{e}.up_proj.weight"] = up.contiguous()
        elif key.endswith("mlp.experts.down_proj"):
            prefix = key[: -len("down_proj")]
            for e in range(val.shape[0]):
                out[f"{prefix}{e}.down_proj.weight"] = val[e].contiguous()
        else:
            out[key] = val
    return {k: v.float().contiguous() for k, v in out.items()}


def routing_gap_ok(model, ids):
    inputs_store = []

    def pre_hook(_mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        inputs_store.append(h.detach())

    moe = model.model.layers[1].mlp
    handle = moe.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    h = inputs_store[0].reshape(-1, HIDDEN)
    probs = (h @ moe.gate.weight.t()).softmax(dim=-1)
    sorted_p, _ = probs.sort(dim=-1, descending=True)
    gap = (sorted_p[:, TOP_K - 1] - sorted_p[:, TOP_K]).min().item()
    return gap


def main():
    model = build_model()

    # Assert the override actually diverges from the YaRN default, otherwise
    # the fixture would not exercise the new code path.
    rot = model.model.rotary_emb
    expected = (yarn_get_mscale(YARN_FACTOR, MSCALE) /
                yarn_get_mscale(YARN_FACTOR, MSCALE_ALL_DIM))
    default = 0.1 * math.log(YARN_FACTOR) + 1.0
    print(f"HF attention_scaling: {rot.attention_scaling:.10f}")
    print(f"mscale formula:       {expected:.10f}")
    print(f"YaRN default:         {default:.10f}")
    assert abs(rot.attention_scaling - expected) < 1e-12, \
        "HF attention_scaling does not match the mscale formula"
    assert abs(expected - default) > 1e-2, \
        "override coincides with default - fixture would not test the override"

    gen = torch.Generator().manual_seed(SEED + 2)
    ids = torch.randint(0, VOCAB, (N_SEQ, MAX_POS), generator=gen)

    gap = routing_gap_ok(model, ids)
    print(f"min top-k routing gap: {gap:.6f}")
    assert gap > 1e-3, "routing gap too small - float32 could flip top-k"

    with torch.no_grad():
        logits = model(input_ids=ids).logits  # float64

    sd = model.state_dict()
    shared_w = sd["model.layers.1.mlp.shared_experts.down_proj.weight"]
    assert shared_w.abs().sum() > 1.0, "shared expert weights ~0"
    krope = sd["model.layers.1.self_attn.kv_a_proj_with_mqa.weight"][
        KV_LORA_RANK:]
    assert krope.abs().sum() > 1.0, "rope-K rows ~0"
    assert logits.abs().max() > 1.0, "logits ~0 - re-randomization vacuous"

    tensors = hub_state_dict(model)
    st_path = os.path.join(OUT_DIR, BASE + ".safetensors")
    save_file(tensors, st_path)
    cfg_path = os.path.join(OUT_DIR, BASE + "_config.json")
    with open(cfg_path, "w") as f:
        json.dump(CONFIG_JSON, f, indent=1)

    reload_dir = os.path.join("/tmp", BASE + "_roundtrip")
    os.makedirs(reload_dir, exist_ok=True)
    os.replace(st_path, os.path.join(reload_dir, "model.safetensors"))
    with open(os.path.join(reload_dir, "config.json"), "w") as f:
        json.dump(CONFIG_JSON, f)
    reloaded = DeepseekV2ForCausalLM.from_pretrained(reload_dir)
    reloaded.config._attn_implementation = "eager"
    reloaded.config._experts_implementation = "eager"
    reloaded.double().eval()
    with torch.no_grad():
        relogits = reloaded(input_ids=ids).logits
    rt_diff = (relogits - logits).abs().max().item()
    print(f"hub-layout round-trip max |diff|: {rt_diff:.3e}")
    assert rt_diff < 1e-12, "hub-layout export does not round-trip"
    os.replace(os.path.join(reload_dir, "model.safetensors"), st_path)

    oracle = {
        "config": {k: v for k, v in CONFIG_JSON.items()},
        "sequences": ids.tolist(),
        "logits": logits.tolist(),
    }
    logits_path = os.path.join(OUT_DIR, BASE + "_logits.json")
    with open(logits_path, "w") as f:
        json.dump(oracle, f)
    print("wrote:")
    for p in (st_path, cfg_path, logits_path):
        print(f"  {p} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
