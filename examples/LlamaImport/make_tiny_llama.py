#!/usr/bin/env python3
"""Construct a RANDOM tiny LlamaForCausalLM with HF transformers and save it
(safetensors + config.json) for parity testing - no download needed.

Coded by Claude (AI).

Usage:
  python3 make_tiny_llama.py <out_dir> [--tie]

Defaults: 2 layers, hidden 64, 4 heads, 2 kv heads (GQA), intermediate 128,
vocab 256, max_pos 64, rope_theta 10000, untied lm_head (--tie ties it).
"""
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: make_tiny_llama.py <out_dir> [--tie]')
    out_dir = sys.argv[1]
    tie = '--tie' in sys.argv[2:]
    torch.manual_seed(424242)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=tie,
        attention_bias=False,
        mlp_bias=False,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    model.save_pretrained(out_dir, safe_serialization=True)
    print(f'wrote random tiny Llama (tie={tie}) to {out_dir}/')
    print('files: model.safetensors, config.json')


if __name__ == '__main__':
    main()
