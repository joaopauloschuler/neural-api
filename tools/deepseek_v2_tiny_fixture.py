#!/usr/bin/env python3
"""Tiny DeepSeek-V2 fixture generator (HF transformers float64 oracle).

Coded by Claude (AI).

Run with the shared venv (torch + transformers, CPU):
    /home/bpsa/x/bin/python tools/deepseek_v2_tiny_fixture.py

Generates:
  tests/fixtures/tiny_deepseek_v2.safetensors - a random DeepSeek-V2-shaped
      checkpoint in the ORIGINAL HUB layout (per-expert
      mlp.experts.{e}.gate_proj/up_proj/down_proj nn.Linear [out, in]
      tensors - transformers >= 5 stores experts PACKED as 3-D
      mlp.experts.gate_up_proj/down_proj parameters, so this script unpacks
      them back to the layout real DeepSeek-V2 checkpoints use, which is
      what neural/neuralpretrained.pas BuildDeepSeekV2FromSafeTensors
      reads). 2 layers so BOTH paths are exercised: layer 0 keeps a DENSE
      SwiGLU MLP (first_k_dense_replace=1) and layer 1 is a DeepSeekMoE
      block (2 shared + 4 routed experts, top-2 softmax gating,
      norm_topk_prob=false, routed_scaling_factor=1.5 so the down_proj
      fold is non-vacuous). MLA dims are chosen so qk_nope_head_dim (6)
      differs from hidden_size/num_attention_heads (4) - the MLA head
      width is NOT tied to the residual width.
  tests/fixtures/tiny_deepseek_v2_config.json - the matching HF config.
  tests/fixtures/tiny_deepseek_v2_logits.json - float64 oracle logits for 3
      fixed token sequences, computed by transformers'
      DeepseekV2ForCausalLM (builtin since v5; eager attention) run in
      double precision.

Weights are RE-RANDOMIZED at O(1) scale (the HF std-0.02 init makes parity
checks vacuous - the activations collapse toward the residual stream; the
ModernBERT fixture learned this). The script also checks a per-token
routing-probability GAP between the k-th and (k+1)-th routed expert so the
float32 Pascal forward cannot flip the top-k selection, and asserts the
shared-expert branch and the rope slices contribute (non-vacuity at the
oracle side; the Pascal test re-asserts on its side).
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers.models.deepseek_v2 import (DeepseekV2Config,
                                             DeepseekV2ForCausalLM)

SEED = 20260612
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "tests", "fixtures")
BASE = "tiny_deepseek_v2"

N_LAYER = 2
N_HEAD = 2
HIDDEN = 8
QK_NOPE = 6           # deliberately != HIDDEN // N_HEAD
QK_ROPE = 4           # decoupled-RoPE width (even)
V_HEAD = 6            # must equal QK_NOPE (importer constraint, true for V2)
KV_LORA_RANK = 5
INTERMEDIATE = 12     # dense MLP width (layer 0)
MOE_INTERMEDIATE = 5  # per-expert width
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
    "rope_scaling": None,
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
    # the grouped_mm experts kernel rejects float64; force the eager loop
    model.config._experts_implementation = "eager"
    # O(1)-scale re-randomization: HF's std-0.02 init leaves every linear
    # output ~0 and the parity check would only measure norms+residual.
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
    """state_dict in the ORIGINAL hub layout: unpack the 3-D packed expert
    parameters (transformers >= 5) into per-expert nn.Linear tensors.
    Builtin packing: gate_up_proj[e] = cat([gate_proj, up_proj]) (gate
    FIRST: forward chunks (gate, up)); down_proj[e] = down_proj."""
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
    """The float32 Pascal forward must not flip the top-k expert choice:
    require a clear per-token gap between routed softmax probs k and k+1.
    The gate nn.Module's forward is bypassed by the MoE block (it calls
    F.linear on gate.weight directly), so hook the MoE module's INPUT and
    recompute the router probabilities from it."""
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
    gen = torch.Generator().manual_seed(SEED + 2)
    ids = torch.randint(0, VOCAB, (N_SEQ, MAX_POS), generator=gen)

    gap = routing_gap_ok(model, ids)
    print(f"min top-k routing gap: {gap:.6f}")
    assert gap > 1e-3, "routing gap too small - float32 could flip top-k"

    with torch.no_grad():
        logits = model(input_ids=ids).logits  # float64

    # Oracle-side non-vacuity: shared experts and the rope path contribute.
    sd = model.state_dict()
    shared_w = sd["model.layers.1.mlp.shared_experts.down_proj.weight"]
    assert shared_w.abs().sum() > 1.0, "shared expert weights ~0"
    krope = sd["model.layers.1.self_attn.kv_a_proj_with_mqa.weight"][
        KV_LORA_RANK:]
    assert krope.abs().sum() > 1.0, "rope-K rows ~0"
    assert logits.abs().max() > 1.0, "logits ~0 - re-randomization vacuous"

    # Hub-layout safetensors + config.
    tensors = hub_state_dict(model)
    st_path = os.path.join(OUT_DIR, BASE + ".safetensors")
    save_file(tensors, st_path)
    cfg_path = os.path.join(OUT_DIR, BASE + "_config.json")
    with open(cfg_path, "w") as f:
        json.dump(CONFIG_JSON, f, indent=1)

    # Round-trip sanity: transformers itself must accept the hub layout
    # (proves the unpacked naming is the real checkpoint convention).
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
