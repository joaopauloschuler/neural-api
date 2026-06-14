#!/usr/bin/env python3
"""Generate a tiny RANDOM Nemotron-H-MoE parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

This is the MoE follow-up to make_pico_nemotronh_fixture.py. It exercises the
'E' (Mixture-of-Experts) block type on the Nemotron-H schedule, which the dense
fixture does NOT cover. The pico schedule "M*E-" carries ALL FOUR block types:
'M'=Mamba-2 mixer, '*'=attention, 'E'=MoE FFN, '-'=non-gated relu2 MLP - so the
new 'E' wiring is verified without losing coverage of the dense blocks.

The 'E' block (HF NemotronHMoE) is a DeepSeek-V3-style sparse FFN that is
distinct from the Mixtral/Jamba MoE already imported:
  - SIGMOID router (NOT softmax): router_logits = gate.weight @ x, then
    .sigmoid();
  - top-k SELECTION on (sigmoid + e_score_correction_bias) but COMBINE weights
    are the RAW sigmoid values, gathered at the selected indices, renormalized
    to sum 1 IFF norm_topk_prob, then scaled by routed_scaling_factor;
  - NON-GATED relu2 experts: down(ReLU(up(x))**2), NO gate proj (the sparse
    counterpart of the dense '-' MLP block) - unlike Mixtral's SwiGLU experts;
  - an ALWAYS-ON shared expert (its own non-gated relu2 MLP at
    moe_shared_expert_intermediate_size) summed into the routed output.

To force a routing-blind import to FAIL, the fixture sets a NON-ZERO
e_score_correction_bias, norm_topk_prob=True, routed_scaling_factor != 1, and a
shared expert, then asserts (below) that EACH of these knobs visibly moves the
oracle logits above the 1e-4 parity gate.

The reference logits are computed by HF transformers in float64 on the CPU
naive path (the fused mamba/conv kernels are not installed; checked below).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_nemotronh_moe_fixture.py
writes tests/fixtures/tiny_nemotronh_moe{.safetensors,_config.json,_logits.json}.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
)

HIDDEN = 8
VOCAB = 13
PATTERN = "M*E-"          # Mamba-2, attention, MoE, relu2 MLP
# attention
NUM_HEADS = 4
KV_HEADS = 2              # GQA: 2 query heads share each kv head
HEAD_DIM = 3             # != HIDDEN//NUM_HEADS (==2): exercise explicit head_dim
# relu2 MLP ('-')
INTER = 12
# MoE ('E')
N_ROUTED_EXPERTS = 4
NUM_EXPERTS_PER_TOK = 2   # top-k routing (< n_routed_experts so masking is real)
MOE_INTER = 6
N_SHARED_EXPERTS = 1
SHARED_INTER = 5          # != MOE_INTER: shared expert has its own width
NORM_TOPK_PROB = True
ROUTED_SCALING_FACTOR = 2.5
# mamba-2 mixer
STATE = 4
MAMBA_HEADS = 2
MAMBA_HEAD_DIM = 4       # d_inner = 2*4 = 8
N_GROUPS = 1            # < mamba_heads: B/C shared across the group's heads
CONV_K = 4
CHUNK = 8
EPS = 1e-5
SEQ_LEN = 7
N_SEQUENCES = 3

torch.manual_seed(20260615)

cfg_dict = {
    'architectures': ['NemotronHForCausalLM'],
    'model_type': 'nemotron_h',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'hybrid_override_pattern': PATTERN,
    'num_attention_heads': NUM_HEADS,
    'num_key_value_heads': KV_HEADS,
    'head_dim': HEAD_DIM,
    'attention_bias': False,
    'intermediate_size': INTER,
    'mlp_hidden_act': 'relu2',
    'mlp_bias': False,
    # MoE
    'n_routed_experts': N_ROUTED_EXPERTS,
    'num_experts_per_tok': NUM_EXPERTS_PER_TOK,
    'moe_intermediate_size': MOE_INTER,
    'n_shared_experts': N_SHARED_EXPERTS,
    'moe_shared_expert_intermediate_size': SHARED_INTER,
    'norm_topk_prob': NORM_TOPK_PROB,
    'routed_scaling_factor': ROUTED_SCALING_FACTOR,
    'n_group': 1,
    'topk_group': 1,
    'moe_latent_size': None,
    # mamba
    'ssm_state_size': STATE,
    'mamba_num_heads': MAMBA_HEADS,
    'mamba_head_dim': MAMBA_HEAD_DIM,
    'n_groups': N_GROUPS,
    'conv_kernel': CONV_K,
    'chunk_size': CHUNK,
    'use_bias': False,
    'use_conv_bias': True,
    'mamba_proj_bias': False,
    'layer_norm_epsilon': EPS,
    'tie_word_embeddings': False,
    'use_cache': True,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}
config = NemotronHConfig(**cfg_dict)
assert config.num_hidden_layers == len(PATTERN), \
    (config.num_hidden_layers, len(PATTERN))
assert config.layers_block_type == ['mamba', 'attention', 'moe', 'mlp'], \
    config.layers_block_type
model = NemotronHForCausalLM(config)

# Force the EAGER per-expert reference loop (NemotronHExperts.forward, the
# documented down(act(up(x))) naive path), not the fused float32-only
# torch.grouped_mm kernel - the oracle runs in float64. Every NemotronHExperts
# module reads self.config._experts_implementation in its forward dispatch.
for mod in model.modules():
    if hasattr(mod, '_experts_implementation'):
        mod._experts_implementation = 'eager'
    cfg = getattr(mod, 'config', None)
    if cfg is not None and hasattr(cfg, '_experts_implementation'):
        cfg._experts_implementation = 'eager'

# The oracle MUST run HF's naive CPU path, not a fused kernel.
assert not any((modeling_nemotron_h.selective_state_update,
                modeling_nemotron_h.causal_conv1d_fn,
                modeling_nemotron_h.causal_conv1d_update)), \
    'CUDA fast-path kernels installed - oracle would not be the naive path'
assert model.model.embeddings.weight.device.type == 'cpu'

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]


def block_type(i):
    return config.layers_block_type[i]


# Re-randomize away from the HF std-0.02 init (vacuous at pico width) to O(1)
# scale so every block type and quirk is visible in the oracle.
with torch.no_grad():
    model.model.embeddings.weight.normal_(0.0, 0.8)
    model.model.norm_f.weight.normal_(1.0, 0.25)
    if model.lm_head.weight.data_ptr() != \
            model.model.embeddings.weight.data_ptr():
        model.lm_head.weight.normal_(0.0, 0.5)
    for i, layer in enumerate(model.model.layers):
        layer.norm.weight.normal_(1.0, 0.25)
        mx = layer.mixer
        bt = block_type(i)
        if bt == 'mamba':
            mx.A_log.copy_(torch.empty_like(mx.A_log).uniform_(-1.0, 1.5))
            mx.D.normal_(0.0, 0.8)
            mx.dt_bias.normal_(0.0, 0.8)
            mx.conv1d.weight.normal_(0.0, 0.5)
            mx.conv1d.bias.normal_(0.0, 0.4)
            mx.in_proj.weight.normal_(0.0, 0.5)
            mx.norm.weight.normal_(1.0, 0.25)
            mx.out_proj.weight.normal_(0.0, 0.5)
        elif bt == 'attention':
            mx.q_proj.weight.normal_(0.0, 0.5)
            mx.k_proj.weight.normal_(0.0, 0.5)
            mx.v_proj.weight.normal_(0.0, 0.5)
            mx.o_proj.weight.normal_(0.0, 0.5)
        elif bt == 'moe':
            mx.gate.weight.normal_(0.0, 0.7)
            # NON-ZERO selection bias: a routing-blind import that ignores it
            # would route to different experts and fail parity.
            mx.gate.e_score_correction_bias.copy_(
                torch.empty_like(mx.gate.e_score_correction_bias)
                .uniform_(-0.8, 0.8))
            mx.experts.up_proj.normal_(0.0, 0.6)
            mx.experts.down_proj.normal_(0.0, 0.6)
            mx.shared_experts.up_proj.weight.normal_(0.0, 0.6)
            mx.shared_experts.down_proj.weight.normal_(0.0, 0.6)
        elif bt == 'mlp':
            mx.up_proj.weight.normal_(0.0, 0.5)
            mx.down_proj.weight.normal_(0.0, 0.5)
        else:
            raise SystemExit('unexpected block type ' + bt)

model = model.double().eval()

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_nemotronh_moe.safetensors')
with open('tests/fixtures/tiny_nemotronh_moe_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_nemotronh_moe_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_nemotronh_moe.safetensors ({len(sd)} tensors) + config + '
      'oracle')
print('layer schedule:', config.layers_block_type)
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every MoE knob visible in the oracle -------------
base = logits

# 0. All FOUR block types present (M/*/E/-).
kinds = set(config.layers_block_type)
assert kinds == {'mamba', 'attention', 'moe', 'mlp'}, kinds
print('all four block types present:', sorted(kinds))

# 1. ROUTING matters: zeroing the e_score_correction_bias changes which experts
# are selected (because the bias is non-zero) and so moves the logits. A
# routing-blind import that drops the bias would match THIS perturbed model, so
# it cannot pass the real parity gate.
nobias = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(nobias.model.layers):
        if block_type(i) == 'moe':
            layer.mixer.gate.e_score_correction_bias.zero_()
d = (run(nobias) - base).abs().max().item()
assert d > 1e-2, f'e_score_correction_bias had no effect on routing ({d})'
print(f'e_score_correction_bias (routing) effect: max |diff| = {d:.4f}')

# 2. norm_topk_prob matters: turning it off changes the combine weights.
nonorm = copy.deepcopy(model)
nonorm.config.norm_topk_prob = False
for i, layer in enumerate(nonorm.model.layers):
    if block_type(i) == 'moe':
        layer.mixer.norm_topk_prob = False
d = (run(nonorm) - base).abs().max().item()
assert d > 1e-2, f'norm_topk_prob had no effect ({d})'
print(f'norm_topk_prob effect: max |diff| = {d:.4f}')

# 3. routed_scaling_factor matters: a different scale rescales the routed sum.
rescale = copy.deepcopy(model)
for i, layer in enumerate(rescale.model.layers):
    if block_type(i) == 'moe':
        layer.mixer.routed_scaling_factor = 1.0
d = (run(rescale) - base).abs().max().item()
assert d > 1e-2, f'routed_scaling_factor had no effect ({d})'
print(f'routed_scaling_factor effect: max |diff| = {d:.4f}')

# 4. Shared expert matters: zeroing its down_proj kills the shared path.
noshared = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(noshared.model.layers):
        if block_type(i) == 'moe':
            layer.mixer.shared_experts.down_proj.weight.zero_()
d = (run(noshared) - base).abs().max().item()
assert d > 1e-2, f'shared expert had no effect ({d})'
print(f'shared expert effect: max |diff| = {d:.4f}')

# 5. Routed experts matter: zeroing all routed up_proj kills the routed path.
norouted = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(norouted.model.layers):
        if block_type(i) == 'moe':
            layer.mixer.experts.up_proj.zero_()
d = (run(norouted) - base).abs().max().item()
assert d > 1e-2, f'routed experts had no effect ({d})'
print(f'routed experts effect: max |diff| = {d:.4f}')

# 6. Experts are NON-GATED relu2 (down(ReLU(up(x))**2)), not SwiGLU. The MoE
# expert module exposes no gate_proj.
for i, layer in enumerate(model.model.layers):
    if block_type(i) == 'moe':
        assert not hasattr(layer.mixer.experts, 'gate_proj')
        assert hasattr(layer.mixer.experts, 'up_proj')
        assert hasattr(layer.mixer.experts, 'down_proj')
act = modeling_nemotron_h.ACT2FN['relu2']
probe = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
assert torch.allclose(act(probe), torch.relu(probe) ** 2)
print('non-gated relu2 experts confirmed')

# 7. top-k masking is real (k < n_routed_experts) and norm/scale/shared set.
assert NUM_EXPERTS_PER_TOK < N_ROUTED_EXPERTS
assert NORM_TOPK_PROB and ROUTED_SCALING_FACTOR != 1.0 and N_SHARED_EXPERTS == 1
assert SHARED_INTER != MOE_INTER
print('top-k masking + distinct shared width confirmed')
print('all fixture self-checks passed')
