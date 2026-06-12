#!/usr/bin/env python3
"""Generate tiny RANDOM T5 / Flan-T5 parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the models are
randomly initialized from pico configs, never downloaded).

Two fixtures, KB-scale, pinned in tests/fixtures/:

  tiny_flan_t5.*: T5ForConditionalGeneration with the Flan-T5 / T5-v1.1
      traits (feed_forward_proj "gated-gelu" i.e. is_gated_act with
      gelu_new wi_0/wi_1, tie_word_embeddings FALSE with a separate
      lm_head, NO d_model**-0.5 logit scaling) plus the quirks every T5
      shares:
        - NO positional embedding; T5 bucketed relative-position bias,
          computed only by block 0 of each stack and SHARED across the
          stack's layers (bidirectional buckets in the encoder, causal
          buckets in the decoder, separate enc/dec tables);
        - attention WITHOUT the 1/sqrt(d_k) scaling (folded into the
          init at pretraining time);
        - a DECOUPLED d_kv (num_heads*d_kv = 6 != d_model = 8);
        - T5LayerNorm = scale-only RMSNorm (no mean subtraction, no
          bias), applied BEFORE every sublayer (pre-norm) + a final norm
          per stack;
        - nn.Linear without biases EVERYWHERE.
  tiny_t5v10.*: same pico shape but the ORIGINAL T5 v1.0 recipe:
      feed_forward_proj "relu" (single wi) and tie_word_embeddings TRUE
      (lm_head = shared embedding, decoder output scaled by
      d_model**-0.5 before the head).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the encoder final hidden states
plus the full encoder-decoder logits for pinned encoder/decoder input
ids.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/t5_tiny_fixture.py
writes tests/fixtures/tiny_flan_t5{.safetensors,_config.json,_logits.json}
   and tests/fixtures/tiny_t5v10{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import T5Config, T5ForConditionalGeneration

N_LAYER = 2
N_HEAD = 2
D_KV = 3                      # decoupled: N_HEAD * D_KV = 6 != D_MODEL = 8
D_MODEL = 8
D_FF = 16
VOCAB = 13
N_BUCKETS = 8
MAX_DISTANCE = 8              # small so ENC_LEN=10 exercises the log buckets
ENC_LEN = 10
DEC_LEN = 6
N_SEQUENCES = 2

torch.manual_seed(20260612)

# Pinned inputs: encoder ids and decoder ids (decoder rows start with the
# decoder_start_token_id 0, the T5 convention).
enc_sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
                  for i in range(ENC_LEN)] for s in range(N_SEQUENCES)]
dec_sequences = [[0] + [(3 * i + 2 * s + 1) % (VOCAB - 1) + 1
                        for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]


def build(name, feed_forward_proj, tie):
    cfg_dict = {
        'architectures': ['T5ForConditionalGeneration'],
        'model_type': 't5',
        'd_model': D_MODEL,
        'd_kv': D_KV,
        'd_ff': D_FF,
        'num_layers': N_LAYER,
        'num_decoder_layers': N_LAYER,
        'num_heads': N_HEAD,
        'relative_attention_num_buckets': N_BUCKETS,
        'relative_attention_max_distance': MAX_DISTANCE,
        'vocab_size': VOCAB,
        'dropout_rate': 0.0,
        'layer_norm_epsilon': 1e-6,
        'feed_forward_proj': feed_forward_proj,
        'tie_word_embeddings': tie,
        'pad_token_id': 0,
        'eos_token_id': 1,
        'decoder_start_token_id': 0,
    }
    model = T5ForConditionalGeneration(
        T5Config(**cfg_dict, attn_implementation='eager'))
    # HF inits with tiny stds at this pico width; boost so every quirk is
    # visible in the oracle: O(1) UNSCALED attention scores, a nonzero
    # relative-position bias, FFN pre-activations in the region where
    # gelu_new and relu/exact-gelu genuinely differ.
    with torch.no_grad():
        for stack in (model.encoder, model.decoder):
            for block in stack.block:
                for layer in block.layer:
                    attn = getattr(layer, 'SelfAttention',
                                   getattr(layer, 'EncDecAttention', None))
                    if attn is not None:
                        attn.q.weight.normal_(0.0, 0.55)
                        attn.k.weight.normal_(0.0, 0.55)
                        attn.v.weight.normal_(0.0, 0.45)
                        attn.o.weight.normal_(0.0, 0.45)
                        if attn.has_relative_attention_bias:
                            attn.relative_attention_bias.weight.normal_(
                                0.0, 0.6)
                    ff = getattr(layer, 'DenseReluDense', None)
                    if ff is not None:
                        if hasattr(ff, 'wi'):
                            ff.wi.weight.normal_(0.0, 0.9)
                        else:
                            ff.wi_0.weight.normal_(0.0, 0.9)
                            ff.wi_1.weight.normal_(0.0, 0.7)
                        ff.wo.weight.normal_(0.0, 0.4)
                # the pre-sublayer norm gains: non-one so they matter
                for layer in block.layer:
                    layer.layer_norm.weight.normal_(1.0, 0.25)
            stack.final_layer_norm.weight.normal_(1.0, 0.25)
        if not tie:
            model.lm_head.weight.normal_(0.0, 0.6)
    model = model.double().eval()

    # Real HF checkpoints drop the tied duplicates (_tied_weights_keys):
    # encoder/decoder embed_tokens always alias shared, and lm_head does
    # too when tie_word_embeddings.
    drop = {'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}
    if tie:
        drop.add('lm_head.weight')
    sd = {k: v.to(torch.float32).clone().contiguous()
          for k, v in model.state_dict().items() if k not in drop}
    save_file(sd, f'tests/fixtures/{name}.safetensors')
    with open(f'tests/fixtures/{name}_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)

    enc_hidden, logits = [], []
    with torch.no_grad():
        for es, ds in zip(enc_sequences, dec_sequences):
            enc_ids = torch.tensor([es])
            dec_ids = torch.tensor([ds])
            enc_hidden.append(
                model.encoder(input_ids=enc_ids).last_hidden_state[0]
                .tolist())
            logits.append(
                model(input_ids=enc_ids,
                      decoder_input_ids=dec_ids).logits[0].tolist())
    with open(f'tests/fixtures/{name}_logits.json', 'w') as f:
        json.dump({'enc_sequences': enc_sequences,
                   'dec_sequences': dec_sequences,
                   'enc_hidden': enc_hidden, 'logits': logits}, f)
    print(f'wrote {name}.safetensors ({len(sd)} tensors) + config + oracle')
    for k in sorted(sd):
        print(f'  {k} {list(sd[k].shape)}')
    return model, cfg_dict, logits


flan, flan_cfg, flan_logits = build('tiny_flan_t5', 'gated-gelu', False)
v10, v10_cfg, v10_logits = build('tiny_t5v10', 'relu', True)

# ---- fixture self-checks: every quirk must be visible in the oracle ----
enc_ids = torch.tensor([enc_sequences[0]])
dec_ids = torch.tensor([dec_sequences[0]])
with torch.no_grad():
    base = flan(input_ids=enc_ids, decoder_input_ids=dec_ids).logits

    # 1. The relative-position bias must matter (zeroing both stacks'
    # tables moves the logits) - a no-bias import FAILS parity.
    import copy
    nobias = copy.deepcopy(flan)
    nobias.encoder.block[0].layer[0].SelfAttention \
        .relative_attention_bias.weight.zero_()
    nobias.decoder.block[0].layer[0].SelfAttention \
        .relative_attention_bias.weight.zero_()
    d = (nobias(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'relpos bias had no effect ({d})'
    print(f'relpos-bias effect on logits: max |diff| = {d:.4f}')

    # 2. The bias must be SHARED across layers: layer 1 reuses layer 0's
    # table, so zeroing the (only) table must change layer-1 attention too.
    # Structural in HF (has_relative_attention_bias only on block 0); we
    # assert block 1 carries NO table of its own.
    assert not flan.encoder.block[1].layer[0].SelfAttention \
        .has_relative_attention_bias
    assert not flan.decoder.block[1].layer[0].SelfAttention \
        .has_relative_attention_bias

    # 3. Gated-GELU vs plain ReLU FFN must be distinguishable: the v1.0
    # fixture uses wi/relu, the flan fixture wi_0|wi_1/gelu_new.
    assert hasattr(flan.encoder.block[0].layer[1].DenseReluDense, 'wi_0')
    assert hasattr(v10.encoder.block[0].layer[1].DenseReluDense, 'wi')

    # 4. gelu_new (tanh approx) vs exact-erf gelu must differ on this
    # fixture above the 1e-4 parity gate (so the import must use the tanh
    # form, TNNetGEGLU).
    erf = copy.deepcopy(flan)
    for stack in (erf.encoder, erf.decoder):
        for block in stack.block:
            ff = getattr(block.layer[-1], 'DenseReluDense', None)
            if ff is not None:
                ff.act = torch.nn.GELU(approximate='none').double()
    d = (erf(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-4, f'gelu_new vs erf-gelu indistinguishable ({d})'
    print(f'gelu_new-vs-erf effect on logits: max |diff| = {d:.6f}')

    # 5. The encoder input must reach the logits through cross-attention.
    other = torch.tensor([enc_sequences[1]])
    d = (flan(input_ids=other, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'encoder ids had no effect on logits ({d})'
    print(f'encoder-input effect on logits: max |diff| = {d:.4f}')

    # 6. Decoder self-attention must be CAUSAL: changing a LATER decoder
    # token must not change EARLIER logit rows.
    dec2 = dec_ids.clone()
    dec2[0, -1] = (dec2[0, -1] + 1) % (VOCAB - 1) + 1
    l2 = flan(input_ids=enc_ids, decoder_input_ids=dec2).logits
    d_early = (l2[0, :-1] - base[0, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'
    print('decoder causality check passed')

    # 7. v1.0 tied head: the d_model**-0.5 pre-head scaling must matter
    # (a no-scale import FAILS parity).
    v10_base = v10(input_ids=enc_ids, decoder_input_ids=dec_ids).logits
    noscale = v10.decoder(
        input_ids=dec_ids,
        encoder_hidden_states=v10.encoder(input_ids=enc_ids)
        .last_hidden_state).last_hidden_state @ v10.shared.weight.T
    d = (noscale - v10_base).abs().max().item()
    assert d > 1e-2, f'tied-head d_model**-0.5 scale had no effect ({d})'
    print(f'tied-head scale effect on logits: max |diff| = {d:.4f}')

    # 8. No biases anywhere (the T5 Linear convention).
    for m in list(flan.modules()) + list(v10.modules()):
        if isinstance(m, torch.nn.Linear):
            assert m.bias is None, 'unexpected Linear bias in T5'
    print('bias-free Linear check passed')
print('all fixture self-checks passed')
