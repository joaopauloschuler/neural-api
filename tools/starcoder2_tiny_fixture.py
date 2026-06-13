#!/usr/bin/env python3
"""Generate tiny RANDOM Starcoder2 parity fixtures for
tests/TestNeuralPretrained.pas (no network access: the models are randomly
initialized from a pico config, never downloaded).

Two fixtures, ~10 KB each, pinned in tests/fixtures/:

  tiny_starcoder2.*: Starcoder2ForCausalLM, FULL attention. Exercises the
      three GPT-2-flavoured pieces the RMSNorm/SwiGLU Llama path never uses:
      - biased nn.LayerNorm norms (input_layernorm / post_attention_layernorm
        / model.norm), NOT RMSNorm - the LayerNorm BIAS is re-randomized (HF
        zero-inits it) and the script ASSERTS resetting it to zero changes the
        logits, so the importer's bias loading is genuinely covered;
      - bias=True (use_bias) on q/k/v AND o_proj AND c_fc/c_proj - the o_proj
        bias (the path OLMo-2 rejects) is re-randomized and asserted non-vacuous;
      - a plain two-matrix gelu_pytorch_tanh FFN (c_fc -> GELU -> c_proj).
      GQA (1 kv head < 2 query heads) and an UNTIED lm_head are on.

  tiny_starcoder2_window.*: identical but with sliding_window set < the
      sequence length, so the banded causal mask genuinely differs from full
      causal attention. The script ASSERTS the windowed logits differ from the
      full-attention logits of the SAME weights.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/starcoder2_tiny_fixture.py
writes tests/fixtures/tiny_starcoder2{,_window}{.safetensors,_config.json,
_logits.json}. Needs torch + transformers (>= the Starcoder2 release) +
safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Starcoder2Config, Starcoder2ForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
D_FF = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
WINDOW = 2

torch.manual_seed(20260613)


def make_cfg(sliding_window):
    return {
        'architectures': ['Starcoder2ForCausalLM'],
        'model_type': 'starcoder2',
        'hidden_size': D_MODEL,
        'intermediate_size': D_FF,
        'num_hidden_layers': N_LAYER,
        'num_attention_heads': N_HEAD,
        'num_key_value_heads': N_KV_HEAD,
        'vocab_size': VOCAB,
        'max_position_embeddings': MAX_POS,
        'norm_epsilon': 1e-5,
        'rope_theta': 100000.0,
        'sliding_window': sliding_window,
        'use_bias': True,
        'tie_word_embeddings': False,
        'hidden_act': 'gelu_pytorch_tanh',
    }


def build(model):
    # HF zero-inits every nn.LayerNorm bias, which would make the bias-loading
    # path vacuous. Re-randomize the norm BIASES (and gains, for good measure)
    # plus the o_proj bias (the OLMo-2-rejected path).
    with torch.no_grad():
        # Larger embeddings + q/k projections sharpen the attention so distant
        # tokens genuinely matter (otherwise a tiny sliding window has almost no
        # effect on the logits and the windowed self-check is vacuous).
        model.model.embed_tokens.weight.normal_(0.0, 1.5)
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.normal_(0.0, 1.0)
            layer.self_attn.k_proj.weight.normal_(0.0, 1.0)
            for ln in (layer.input_layernorm, layer.post_attention_layernorm):
                ln.weight.normal_(1.0, 0.3)
                ln.bias.normal_(0.0, 0.3)
            layer.self_attn.o_proj.bias.normal_(0.0, 0.3)
        model.model.norm.weight.normal_(1.0, 0.3)
        model.model.norm.bias.normal_(0.0, 0.3)
    return model.double().eval()


sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
seq_t = [torch.tensor([seq]) for seq in sequences]

# ---- shared random weights, two configs (full vs windowed) ----
base_model = build(Starcoder2ForCausalLM(
    Starcoder2Config(**make_cfg(None), attn_implementation='eager')))
shared_sd = {k: v.clone() for k, v in base_model.state_dict().items()}


def write_fixture(name, sliding_window):
    cfg = make_cfg(sliding_window)
    model = Starcoder2ForCausalLM(
        Starcoder2Config(**cfg, attn_implementation='eager'))
    model.load_state_dict({k: v.clone() for k, v in shared_sd.items()})
    model = model.double().eval()
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, f'tests/fixtures/{name}.safetensors')
    with open(f'tests/fixtures/{name}_config.json', 'w') as f:
        json.dump(cfg, f, indent=1)
    with torch.no_grad():
        logits = [model(input_ids=t).logits[0].tolist() for t in seq_t]
    with open(f'tests/fixtures/{name}_logits.json', 'w') as f:
        json.dump({'sequences': sequences, 'logits': logits}, f)
    print(f'wrote {name}.safetensors ({len(sd)} tensors) + config + logits')
    return model


full_model = write_fixture('tiny_starcoder2', None)
window_model = write_fixture('tiny_starcoder2_window', WINDOW)

# ---- self-checks (the deltas must be non-vacuous) ----
with torch.no_grad():
    base = full_model(input_ids=seq_t[0]).logits


def logits_with(sd_patch):
    patched = Starcoder2ForCausalLM(
        Starcoder2Config(**make_cfg(None), attn_implementation='eager'))
    sd_full = {k: v.clone() for k, v in full_model.state_dict().items()}
    sd_full.update(sd_patch)
    patched.load_state_dict(sd_full)
    patched = patched.double().eval()
    with torch.no_grad():
        return patched(input_ids=seq_t[0]).logits


# Zeroing the LayerNorm biases MUST change the logits.
ln_bias_zero = {k: torch.zeros_like(v)
                for k, v in full_model.state_dict().items()
                if k.endswith('layernorm.bias') or k == 'model.norm.bias'}
ln_effect = (base - logits_with(ln_bias_zero)).abs().max().item()
assert ln_effect > 1e-3, \
    f'LayerNorm biases had no effect on the logits ({ln_effect})'
print(f'LayerNorm-bias effect: max |diff| = {ln_effect:.4f}')

# Zeroing the o_proj biases MUST change the logits (the OLMo-2-rejected path).
o_bias_zero = {k: torch.zeros_like(v)
               for k, v in full_model.state_dict().items()
               if k.endswith('o_proj.bias')}
o_effect = (base - logits_with(o_bias_zero)).abs().max().item()
assert o_effect > 1e-3, \
    f'o_proj biases had no effect on the logits ({o_effect})'
print(f'o_proj-bias effect: max |diff| = {o_effect:.4f}')

# The sliding window MUST change the logits vs full attention (same weights).
with torch.no_grad():
    full_l = full_model(input_ids=seq_t[0]).logits
    win_l = window_model(input_ids=seq_t[0]).logits
win_effect = (full_l - win_l).abs().max().item()
assert win_effect > 1e-3, \
    f'sliding window had no effect on the logits ({win_effect})'
print(f'sliding-window effect: max |diff| = {win_effect:.4f}')
