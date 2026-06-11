#!/usr/bin/env python3
"""Real-weight parity check: CAI GPT-2 importer vs HuggingFace transformers.

Coded by Claude (AI).

Loads a GPT-2 safetensors checkpoint (full or sliced by slice_gpt2.py) into
transformers' GPT2LMHeadModel with a config inferred from the tensor shapes,
runs the SAME token ids as a Pascal GPT2LogitsDump run, and diffs every
logit of every position against the Pascal JSON.

Usage:
  bin/.../GPT2LogitsDump model.safetensors 16 0 464 262 976 > /tmp/cai.json
  python3 compare_hf_logits.py model.safetensors /tmp/cai.json

Both sides must see the same checkpoint. Exits non-zero if max |diff|
exceeds the gate (1e-3; observed parity should be ~1e-4 in f32).
"""
import json
import sys

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

GATE = 1e-3


def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: compare_hf_logits.py <model.safetensors> <cai_logits.json>')
    ckpt_path, cai_path = sys.argv[1], sys.argv[2]

    state = load_file(ckpt_path)
    # The openai-community/gpt2 file has no prefix; GPT2LMHeadModel exports
    # carry "transformer.". Normalize to unprefixed GPT2Model keys.
    state = {(k[len('transformer.'):] if k.startswith('transformer.') else k): v
             for k, v in state.items() if k != 'lm_head.weight'}

    vocab, n_embd = state['wte.weight'].shape
    n_ctx = state['wpe.weight'].shape[0]
    n_layer = sum(1 for k in state if k.endswith('.ln_1.weight'))
    config = GPT2Config(vocab_size=vocab, n_positions=n_ctx, n_embd=n_embd,
                        n_layer=n_layer, n_head=n_embd // 64)
    print(f'HF config: n_layer={n_layer}, n_head={config.n_head}, '
          f'n_embd={n_embd}, n_ctx={n_ctx}, vocab={vocab}')

    model = GPT2LMHeadModel(config)
    missing, unexpected = model.transformer.load_state_dict(state, strict=False)
    # Only the non-persistent causal-mask buffers may be absent.
    bad = [k for k in missing if '.attn.bias' not in k and
           '.attn.masked_bias' not in k]
    if bad or unexpected:
        sys.exit(f'state_dict mismatch: missing={bad} unexpected={unexpected}')
    model.tie_weights()  # lm_head := wte (same tying the Pascal importer copies)
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
