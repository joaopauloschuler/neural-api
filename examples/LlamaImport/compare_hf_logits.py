#!/usr/bin/env python3
"""Real-weight parity check: CAI Llama importer vs HuggingFace transformers.

Coded by Claude (AI).

Loads a Llama-architecture safetensors checkpoint (random tiny model from
make_tiny_llama.py, or a real model sliced by slice_llama.py) into
transformers' LlamaForCausalLM using the config.json next to it, runs the
SAME token ids as a Pascal LlamaLogitsDump run, and diffs every logit of
every position against the Pascal JSON.

Usage:
  bin/.../LlamaLogitsDump dir/model.safetensors 16 1 5 99 > /tmp/cai.json
  python3 compare_hf_logits.py dir/model.safetensors /tmp/cai.json

Both sides must see the same checkpoint. Exits non-zero if max |diff|
exceeds the gate (1e-3; observed parity should be ~1e-4 in f32).
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

GATE = 1e-3


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: compare_hf_logits.py <model.safetensors> <cai_logits.json>')
    ckpt_path, cai_path = sys.argv[1], sys.argv[2]
    cfg_path = os.path.join(os.path.dirname(ckpt_path) or '.', 'config.json')

    cfg = json.load(open(cfg_path))
    config = LlamaConfig(**{k: v for k, v in cfg.items()
                            if k not in ('architectures', 'model_type',
                                         'torch_dtype', 'transformers_version')})
    print(f'HF config: layers={config.num_hidden_layers}, '
          f'heads={config.num_attention_heads}, '
          f'kv_heads={config.num_key_value_heads}, '
          f'hidden={config.hidden_size}, vocab={config.vocab_size}, '
          f'tied={config.tie_word_embeddings}')

    state = load_file(ckpt_path)
    # Older exports serialize the per-layer RoPE inv_freq buffers; modern
    # transformers keeps them non-persistent (the Pascal importer ignores
    # them too - RoPE is structural on both sides).
    state = {k: v for k, v in state.items() if 'rotary_emb.inv_freq' not in k}

    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if config.tie_word_embeddings:
        missing = [k for k in missing if k != 'lm_head.weight']
        model.tie_weights()
    if missing or unexpected:
        sys.exit(f'state_dict mismatch: missing={missing} '
                 f'unexpected={unexpected}')
    model = model.float()
    model.eval()

    with open(cai_path) as f:
        cai = json.load(f)
    tokens = cai['tokens']
    cai_logits = torch.tensor(cai['logits'], dtype=torch.float32)

    with torch.no_grad():
        hf_logits = model(torch.tensor([tokens])).logits[0]

    if hf_logits.shape != cai_logits.shape:
        sys.exit(f'shape mismatch: HF {tuple(hf_logits.shape)} vs '
                 f'CAI {tuple(cai_logits.shape)}')

    diff = (hf_logits - cai_logits).abs()
    per_pos = diff.max(dim=1).values
    for i, (tok, d) in enumerate(zip(tokens, per_pos.tolist())):
        agree = torch.argmax(hf_logits[i]) == torch.argmax(cai_logits[i])
        print(f'pos {i:2d} (token {tok:5d}): max|diff|={d:.3e}  '
              f'argmax {"agrees" if agree else "DISAGREES"}')
    max_diff = diff.max().item()
    print(f'logit range: [{hf_logits.min():.2f}, {hf_logits.max():.2f}]')
    print(f'max |diff| over all {diff.numel()} logits: {max_diff:.3e} '
          f'(gate {GATE:g})')
    if max_diff >= GATE:
        sys.exit('PARITY FAILURE')
    print('PARITY OK')


if __name__ == '__main__':
    main()
