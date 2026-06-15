#!/usr/bin/env python3
"""Generate a tiny RANDOM GPT-BigCode / StarCoder-v1 parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

  tiny_gptbigcode.*: GPTBigCodeForCausalLM, MULTI-QUERY attention
      (multi_query=True -> a single shared K/V head, kv_dim = head_dim).
      Exercises the pieces the StarCoder2 path does NOT use:
      - LEARNED absolute position embeddings (wpe table added to wte), the
        GPT-2 path, instead of RoPE; the script ASSERTS the wpe table genuinely
        affects the logits (zeroing it changes them);
      - a FUSED c_attn slab [q (embed_dim) | k (head_dim) | v (head_dim)]
        sliced and split GPT-2/Phi-3 style;
      - MULTI-QUERY attention via the grouped-SDPA path (1 KV head < n query
        heads); the script ASSERTS the single shared K/V head is genuinely
        broadcast (perturbing it changes ALL query heads' outputs).
      Pre-LN blocks with biased nn.LayerNorm, bias=True on every nn.Linear,
      a two-matrix gelu_pytorch_tanh FFN, and a TIED lm_head (= wte).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gptbigcode_tiny_fixture.py
writes tests/fixtures/tiny_gptbigcode{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the GPT-BigCode release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM

N_LAYER = 2
N_HEAD = 2
D_MODEL = 8
D_FF = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260613)


def make_cfg():
    return {
        'architectures': ['GPTBigCodeForCausalLM'],
        'model_type': 'gpt_bigcode',
        'n_embd': D_MODEL,
        'n_inner': D_FF,
        'n_layer': N_LAYER,
        'n_head': N_HEAD,
        'n_positions': MAX_POS,
        'vocab_size': VOCAB,
        'layer_norm_epsilon': 1e-5,
        'multi_query': True,
        'scale_attn_weights': True,
        'tie_word_embeddings': True,
        'activation_function': 'gelu_pytorch_tanh',
        'bos_token_id': 0,
        'eos_token_id': 0,
    }


def build(model):
    # HF zero-inits every nn.LayerNorm bias, which would make the bias-loading
    # path vacuous. Re-randomize the norm BIASES (and gains) plus the attention
    # projection biases. Sharpen the embeddings / q-k so distant positions
    # genuinely matter (makes the learned-position self-check non-vacuous).
    with torch.no_grad():
        model.transformer.wte.weight.normal_(0.0, 1.5)
        model.transformer.wpe.weight.normal_(0.0, 1.5)
        for block in model.transformer.h:
            block.attn.c_attn.weight.normal_(0.0, 1.0)
            block.attn.c_attn.bias.normal_(0.0, 0.3)
            block.attn.c_proj.bias.normal_(0.0, 0.3)
            for ln in (block.ln_1, block.ln_2):
                ln.weight.normal_(1.0, 0.3)
                ln.bias.normal_(0.0, 0.3)
        model.transformer.ln_f.weight.normal_(1.0, 0.3)
        model.transformer.ln_f.bias.normal_(0.0, 0.3)
    return model.double().eval()


sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
seq_t = [torch.tensor([seq]) for seq in sequences]

cfg = make_cfg()
model = build(GPTBigCodeForCausalLM(
    GPTBigCodeConfig(**cfg, attn_implementation='eager')))

sd = {k: v.to(torch.float32).contiguous() for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gptbigcode.safetensors')
with open('tests/fixtures/tiny_gptbigcode_config.json', 'w') as f:
    json.dump(cfg, f, indent=1)
with torch.no_grad():
    logits = [model(input_ids=t).logits[0].tolist() for t in seq_t]
with open('tests/fixtures/tiny_gptbigcode_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gptbigcode.safetensors ({len(sd)} tensors) + config + logits')

# ---- self-checks (the deltas must be non-vacuous) ----
with torch.no_grad():
    base = model(input_ids=seq_t[0]).logits


def logits_with(sd_patch):
    patched = GPTBigCodeForCausalLM(
        GPTBigCodeConfig(**make_cfg(), attn_implementation='eager'))
    sd_full = {k: v.clone() for k, v in model.state_dict().items()}
    sd_full.update(sd_patch)
    patched.load_state_dict(sd_full)
    patched = patched.double().eval()
    with torch.no_grad():
        return patched(input_ids=seq_t[0]).logits


# Zeroing the learned-position table MUST change the logits (no-RoPE path).
wpe_zero = {'transformer.wpe.weight':
            torch.zeros_like(model.transformer.wpe.weight)}
wpe_effect = (base - logits_with(wpe_zero)).abs().max().item()
assert wpe_effect > 1e-3, \
    f'learned-position table had no effect on the logits ({wpe_effect})'
print(f'learned-position effect: max |diff| = {wpe_effect:.4f}')

# Perturbing the SINGLE shared K/V head MUST change the logits (MQA broadcast).
# The K/V rows are the LAST 2*head_dim rows of every c_attn slab.
head_dim = D_MODEL // N_HEAD
kv_patch = {}
for li in range(N_LAYER):
    w = model.transformer.h[li].attn.c_attn.weight.clone()
    w[D_MODEL:, :] += 0.5  # shift the shared K and V rows
    kv_patch[f'transformer.h.{li}.attn.c_attn.weight'] = w
kv_effect = (base - logits_with(kv_patch)).abs().max().item()
assert kv_effect > 1e-3, \
    f'shared K/V head had no effect on the logits ({kv_effect})'
print(f'shared-KV-head effect: max |diff| = {kv_effect:.4f}')
