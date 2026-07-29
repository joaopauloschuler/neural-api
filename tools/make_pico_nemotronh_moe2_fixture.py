#!/usr/bin/env python3
"""Generate tests/fixtures/tiny_nemotronh_moe2.{safetensors,_config.json}.

A pico Nemotron-H checkpoint on schedule "EE" - TWO MoE ('E') blocks, each
with its OWN e_score_correction_bias. The importer must land block 0's bias on
block 0's router gate and block 1's bias on block 1's gate; a builder that
keeps a single gate reference across the architecture pass writes both onto the
last block. The two bias vectors are deliberately disjoint in value so the test
can tell them apart.

Pure stdlib (no torch / transformers): the fixture carries no reference logits,
only the routing biases the import test reads back.
"""

import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, os.pardir, "tests", "fixtures")

HIDDEN = 4
VOCAB = 6
EXPERTS = 3
MOE_INTER = 3
SHARED_INTER = 2

# Block 0 and block 1 selection biases: disjoint value ranges.
BIAS = {0: [1.0, 2.0, 3.0], 1: [-4.0, -5.0, -6.0]}


def ramp(n, start=0.0, step=0.01):
    return [start + step * i for i in range(n)]


def main():
    tensors = {}
    tensors["model.embeddings.weight"] = ([VOCAB, HIDDEN], ramp(VOCAB * HIDDEN, 0.01))
    tensors["lm_head.weight"] = ([VOCAB, HIDDEN], ramp(VOCAB * HIDDEN, -0.02))
    tensors["model.norm_f.weight"] = ([HIDDEN], [1.0] * HIDDEN)
    for layer in (0, 1):
        p = "model.layers.%d." % layer
        tensors[p + "norm.weight"] = ([HIDDEN], [1.0] * HIDDEN)
        tensors[p + "mixer.gate.weight"] = (
            [EXPERTS, HIDDEN], ramp(EXPERTS * HIDDEN, 0.03 * (layer + 1)))
        tensors[p + "mixer.gate.e_score_correction_bias"] = ([EXPERTS], BIAS[layer])
        tensors[p + "mixer.experts.up_proj"] = (
            [EXPERTS, MOE_INTER, HIDDEN], ramp(EXPERTS * MOE_INTER * HIDDEN, 0.02))
        tensors[p + "mixer.experts.down_proj"] = (
            [EXPERTS, HIDDEN, MOE_INTER], ramp(EXPERTS * HIDDEN * MOE_INTER, 0.05))
        tensors[p + "mixer.shared_experts.up_proj.weight"] = (
            [SHARED_INTER, HIDDEN], ramp(SHARED_INTER * HIDDEN, 0.04))
        tensors[p + "mixer.shared_experts.down_proj.weight"] = (
            [HIDDEN, SHARED_INTER], ramp(HIDDEN * SHARED_INTER, 0.06))

    header = {}
    blob = bytearray()
    for name in sorted(tensors):
        shape, values = tensors[name]
        data = struct.pack("<%df" % len(values), *values)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    hbytes = json.dumps(header).encode("utf-8")
    hbytes += b" " * ((8 - len(hbytes) % 8) % 8)
    with open(os.path.join(FIX, "tiny_nemotronh_moe2.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(bytes(blob))

    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "hybrid_override_pattern": "EE",
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "attention_bias": False,
        "intermediate_size": 4,
        "mlp_hidden_act": "relu2",
        "mlp_bias": False,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": MOE_INTER,
        "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": SHARED_INTER,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "n_group": 1,
        "topk_group": 1,
        "moe_latent_size": None,
        "ssm_state_size": 4,
        "mamba_num_heads": 2,
        "mamba_head_dim": 2,
        "n_groups": 1,
        "conv_kernel": 4,
        "chunk_size": 8,
        "use_bias": False,
        "use_conv_bias": True,
        "mamba_proj_bias": False,
        "layer_norm_epsilon": 1e-05,
        "tie_word_embeddings": False,
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    with open(os.path.join(FIX, "tiny_nemotronh_moe2_config.json"), "w") as f:
        json.dump(config, f, indent=1)


if __name__ == "__main__":
    main()
