#!/usr/bin/env python3
"""Generate pico HF PEFT LoRA adapters + merged-model logit oracles for the
LoRA merge in HF tensor space (TNNetLoRAMergeReader / BuildFromPretrained's
pLoRAAdapterDir in neural/neuralpretrained.pas). No network access: the base
weights are the committed tiny fixtures, the adapters are random.

Two fixtures, written by PEFT's own save_pretrained (adapter_model.safetensors
+ adapter_config.json) next to a logits.json oracle:

  tests/fixtures/lora_merge_qwen3_5/: base tiny_qwen3_5 (the Qwen3.5 hybrid,
      checkpoint keys under "model.language_model."). The adapter wraps the
      TEXT model (Qwen3_5TextModel) the way kev does, so its keys are
      "base_model.model.layers.N.<module>.lora_{A,B}.weight", on every Kev
      target module: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
      linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.
      r=2, lora_alpha=3, plus rank_pattern {"down_proj": 3} and alpha_pattern
      {"o_proj": 5} so per-module rank and scale are pinned too.
  tests/fixtures/lora_merge_qwen3/: base tiny_qwen3 (attention only, keys
      under "model."), adapter on q/k/v/o/gate/up/down, r=2, lora_alpha=4.

The oracle is HF float64 logits of merge_and_unload(): the merge itself runs
in float32 (W += (alpha/r) * B @ A, as kev merges), then the merged model is
cast to float64. The sequences are the base fixtures' own.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/lora_merge_tiny_fixture.py
Needs torch + transformers (with qwen3_5) + safetensors + peft.
"""
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import (Qwen3Config, Qwen3ForCausalLM, Qwen3_5ForCausalLM,
                          Qwen3_5TextConfig)

FIXTURES = 'tests/fixtures'
KEV_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj',
               'up_proj', 'down_proj', 'in_proj_qkv', 'in_proj_z',
               'in_proj_a', 'in_proj_b', 'out_proj']


def load_qwen3_5_base():
    with open(os.path.join(FIXTURES, 'tiny_qwen3_5_config.json')) as f:
        text_cfg = json.load(f)['text_config']
    config = Qwen3_5TextConfig(
        **{k: v for k, v in text_cfg.items() if k != 'model_type'})
    model = Qwen3_5ForCausalLM(config)
    model.config._attn_implementation = 'eager'
    sd = {}
    for k, v in load_file(os.path.join(FIXTURES,
                                       'tiny_qwen3_5.safetensors')).items():
        if k.startswith('model.language_model.'):
            sd['model.' + k[len('model.language_model.'):]] = v
        elif k == 'lm_head.weight':
            sd[k] = v
    model.load_state_dict(sd)
    return model.float().eval(), 'tiny_qwen3_5_logits.json'


def load_qwen3_base():
    with open(os.path.join(FIXTURES, 'tiny_qwen3_config.json')) as f:
        cfg = json.load(f)
    model = Qwen3ForCausalLM(Qwen3Config(**cfg, attn_implementation='eager'))
    sd = load_file(os.path.join(FIXTURES, 'tiny_qwen3.safetensors'))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {'lm_head.weight'}, \
        (missing, unexpected)
    model.tie_weights()
    return model.float().eval(), 'tiny_qwen3_logits.json'


def build_fixture(name, load_base, lora_cfg, seed, lora_std):
    torch.manual_seed(seed)
    model, base_logits_file = load_base()
    with open(os.path.join(FIXTURES, base_logits_file)) as f:
        sequences = json.load(f)['sequences']
    with torch.no_grad():
        base_logits = [model.double()(input_ids=torch.tensor([s])).logits[0]
                       for s in sequences]
    model = model.float()
    # PEFT wraps the TEXT model (as kev and Open-Jev do): keys come out as
    # base_model.model.layers.N.<module>.lora_{A,B}.weight.
    peft_model = get_peft_model(model.model, lora_cfg)
    with torch.no_grad():
        for pname, p in peft_model.named_parameters():
            if 'lora_A' in pname or 'lora_B' in pname:
                p.normal_(0.0, lora_std)  # non-zero B: the delta is real
    out_dir = os.path.join(FIXTURES, name)
    peft_model.save_pretrained(out_dir)
    readme = os.path.join(out_dir, 'README.md')
    if os.path.exists(readme):
        os.remove(readme)
    keys = sorted(load_file(os.path.join(out_dir,
                                         'adapter_model.safetensors')))
    assert all(k.startswith('base_model.model.layers.') for k in keys), keys
    model.model = peft_model.merge_and_unload()  # float32 merge
    with torch.no_grad():
        m32 = [model(input_ids=torch.tensor([s])).logits[0].double()
               for s in sequences]
        model = model.double()
        logits = [model(input_ids=torch.tensor([s])).logits[0]
                  for s in sequences]
    effect = max((a - b).abs().max().item()
                 for a, b in zip(logits, base_logits))
    scale = max(l.abs().max().item() for l in logits)
    assert effect > 0.05 * scale, \
        f'{name}: adapter barely moves the logits ({effect} vs {scale})'
    drift = max((a - b).abs().max().item() for a, b in zip(m32, logits))
    assert drift < 3e-5, f'{name}: fixture too hot, f32 drift {drift}'
    with open(os.path.join(out_dir, 'logits.json'), 'w') as f:
        json.dump({'sequences': sequences,
                   'logits': [l.tolist() for l in logits]}, f)
    print(f'{name}: {len(keys)} adapter tensors; merge moves logits by '
          f'{effect:.4f} (max |logit| {scale:.4f}); f32 drift {drift:.2e}')


build_fixture(
    'lora_merge_qwen3_5', load_qwen3_5_base,
    LoraConfig(task_type='FEATURE_EXTRACTION', r=2, lora_alpha=3,
               lora_dropout=0.0, target_modules=KEV_TARGETS,
               rank_pattern={'down_proj': 3}, alpha_pattern={'o_proj': 5}),
    seed=20261020, lora_std=0.1)
build_fixture(
    'lora_merge_qwen3', load_qwen3_base,
    LoraConfig(task_type='FEATURE_EXTRACTION', r=2, lora_alpha=4,
               lora_dropout=0.0,
               target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                               'gate_proj', 'up_proj', 'down_proj']),
    seed=20261019, lora_std=0.3)
