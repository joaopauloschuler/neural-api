#!/usr/bin/env python3
"""Generate a tiny RANDOM BART parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_bart.*: BartForConditionalGeneration (the facebook/bart-* /
      sshleifer/distilbart-* summarization architecture) with every BART
      trait the importer must reproduce:
        - POST-norm encoder-decoder blocks: LayerNorm AFTER the residual
          add (self_attn_layer_norm / encoder_attn_layer_norm /
          final_layer_norm per block; NO final stack norm), plain BIASED
          nn.LayerNorm (eps 1e-5);
        - LEARNED ABSOLUTE position embeddings with BART's +2 padding
          offset: position p reads row p+2 of an (max_pos+2, d_model)
          table (rows 0/1 are the padding-idx slots, never read);
        - a layernorm_embedding LayerNorm AFTER token+position embeddings
          in BOTH the encoder and the decoder (Marian has none);
        - exact-erf GELU FFN (fc2(gelu(fc1(x))), both biased);
        - standard 1/sqrt(head_dim) attention scaling, ALL q/k/v/out
          projections biased;
        - scale_embedding FALSE (no sqrt(d_model) scaling - the BART
          default);
        - shared source/target embedding TIED to the lm_head PLUS the
          final_logits_bias vector added to every logit row;
        - decoder_start_token_id = eos_token_id (BART's shift; rows of the
          pinned decoder ids start with it).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the encoder final hidden states
plus the full encoder-decoder logits for pinned encoder/decoder input
ids.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bart_tiny_fixture.py
writes tests/fixtures/tiny_bart{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import BartConfig, BartForConditionalGeneration

N_LAYER = 2
N_HEAD = 2                    # head_dim = D_MODEL / N_HEAD = 4
D_MODEL = 8
D_FF = 16
VOCAB = 13
PAD = 1                       # BART convention: pad_token_id = 1
BOS = 0
EOS = 2
MAX_POS = 16                  # embed_positions has MAX_POS + 2 rows
ENC_LEN = 10
DEC_LEN = 6
N_SEQUENCES = 2

torch.manual_seed(20260613)

# Pinned inputs: encoder ids are bos ... eos (the BART convention) and
# decoder ids start with decoder_start_token_id = eos (BART's shift).
enc_sequences = [[BOS] + [(5 * i + 3 * s + s * s) % (VOCAB - 3) + 3
                          for i in range(ENC_LEN - 2)] + [EOS]
                 for s in range(N_SEQUENCES)]
dec_sequences = [[EOS] + [(3 * i + 2 * s + 1) % (VOCAB - 3) + 3
                          for i in range(DEC_LEN - 1)]
                 for s in range(N_SEQUENCES)]

cfg_dict = {
    'architectures': ['BartForConditionalGeneration'],
    'model_type': 'bart',
    'd_model': D_MODEL,
    'encoder_layers': N_LAYER,
    'decoder_layers': N_LAYER,
    'encoder_attention_heads': N_HEAD,
    'decoder_attention_heads': N_HEAD,
    'encoder_ffn_dim': D_FF,
    'decoder_ffn_dim': D_FF,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation_function': 'gelu',
    'scale_embedding': False,
    'tie_word_embeddings': True,
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'activation_dropout': 0.0,
    'bos_token_id': BOS,
    'pad_token_id': PAD,
    'eos_token_id': EOS,
    'decoder_start_token_id': EOS,
}
model = BartForConditionalGeneration(
    BartConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with tiny stds at this pico width; boost so every quirk is
# visible in the oracle: O(1) attention scores, FFN pre-activations in the
# region where gelu and relu/swish genuinely differ, layer-norm gains and
# biases away from (1, 0), a final_logits_bias that moves the argmax, and
# learned position rows that genuinely matter.
with torch.no_grad():
    model.model.shared.weight.normal_(0.0, 0.5)
    model.model.encoder.embed_positions.weight.normal_(0.0, 0.5)
    model.model.decoder.embed_positions.weight.normal_(0.0, 0.5)
    for stack in (model.model.encoder, model.model.decoder):
        stack.layernorm_embedding.weight.normal_(1.0, 0.25)
        stack.layernorm_embedding.bias.normal_(0.0, 0.2)
        for layer in stack.layers:
            attns = [layer.self_attn]
            if hasattr(layer, 'encoder_attn'):
                attns.append(layer.encoder_attn)
            for attn in attns:
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj,
                             attn.out_proj):
                    proj.weight.normal_(0.0, 0.45)
                    proj.bias.normal_(0.0, 0.2)
            layer.fc1.weight.normal_(0.0, 0.7)
            layer.fc1.bias.normal_(0.0, 0.3)
            layer.fc2.weight.normal_(0.0, 0.4)
            layer.fc2.bias.normal_(0.0, 0.2)
            norms = [layer.self_attn_layer_norm, layer.final_layer_norm]
            if hasattr(layer, 'encoder_attn_layer_norm'):
                norms.append(layer.encoder_attn_layer_norm)
            for norm in norms:
                norm.weight.normal_(1.0, 0.25)
                norm.bias.normal_(0.0, 0.2)
    model.final_logits_bias.normal_(0.0, 0.6)
model = model.double().eval()

# Real HF checkpoints drop the tied duplicates: encoder/decoder
# embed_tokens and lm_head alias shared (_tied_weights_keys). BART's
# LEARNED embed_positions ARE saved (unlike Marian's static sinusoids).
drop = {
    'model.encoder.embed_tokens.weight',
    'model.decoder.embed_tokens.weight',
    'lm_head.weight',
}
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k not in drop}
save_file(sd, 'tests/fixtures/tiny_bart.safetensors')
with open('tests/fixtures/tiny_bart_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

enc_hidden, logits = [], []
with torch.no_grad():
    for es, ds in zip(enc_sequences, dec_sequences):
        enc_ids = torch.tensor([es])
        dec_ids = torch.tensor([ds])
        enc_hidden.append(
            model.model.encoder(input_ids=enc_ids).last_hidden_state[0]
            .tolist())
        logits.append(
            model(input_ids=enc_ids,
                  decoder_input_ids=dec_ids).logits[0].tolist())
with open('tests/fixtures/tiny_bart_logits.json', 'w') as f:
    json.dump({'enc_sequences': enc_sequences,
               'dec_sequences': dec_sequences,
               'enc_hidden': enc_hidden, 'logits': logits}, f)
print(f'wrote tiny_bart.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
import copy

enc_ids = torch.tensor([enc_sequences[0]])
dec_ids = torch.tensor([dec_sequences[0]])
with torch.no_grad():
    base = model(input_ids=enc_ids, decoder_input_ids=dec_ids).logits

    # 1. embed_positions uses the +2 padding offset: position 0 reads row
    # 2 of the table. Verify HF's BartLearnedPositionalEmbedding offset.
    off = model.model.encoder.embed_positions.offset
    assert off == 2, f'unexpected embed_positions offset ({off})'
    assert (model.model.encoder.embed_positions.weight.shape[0]
            == MAX_POS + 2), 'embed_positions must have max_pos+2 rows'
    print('embed_positions +2 offset check passed')

    # 2. The positions must matter (zeroing the USED rows moves the logits)
    # - an importer that forgets them FAILS parity.
    nopos = copy.deepcopy(model)
    nopos.model.encoder.embed_positions.weight.zero_()
    nopos.model.decoder.embed_positions.weight.zero_()
    d = (nopos(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'positions had no effect ({d})'
    print(f'learned-position effect on logits: max |diff| = {d:.4f}')

    # 3. The +2 OFFSET specifically must matter: rolling the used window by
    # one row (reading rows 1.. instead of 2..) must change the logits, so
    # an importer that drops the offset FAILS.
    shifted = copy.deepcopy(model)
    for stack in (shifted.model.encoder, shifted.model.decoder):
        w = stack.embed_positions.weight.data
        w[2:] = model.model.encoder.embed_positions.weight.data[1:-1] \
            if stack is shifted.model.encoder else \
            model.model.decoder.embed_positions.weight.data[1:-1]
    d = (shifted(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'+2 position offset had no effect ({d})'
    print(f'+2 offset effect on logits: max |diff| = {d:.4f}')

    # 4. layernorm_embedding must matter (a stack that forgets it FAILS).
    nle = copy.deepcopy(model)
    nle.model.encoder.layernorm_embedding.bias += 1.0
    d = (nle(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'layernorm_embedding had no effect ({d})'
    print(f'layernorm_embedding effect on logits: max |diff| = {d:.4f}')

    # 5. gelu vs relu (and vs swish) must be distinguishable above the
    # 1e-4 parity gate - the wrong FFN activation FAILS.
    for wrong in ('relu', 'silu'):
        alt = copy.deepcopy(model)
        for stack in (alt.model.encoder, alt.model.decoder):
            for layer in stack.layers:
                layer.activation_fn = (torch.nn.ReLU() if wrong == 'relu'
                                       else torch.nn.SiLU())
        d = (alt(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
             base).abs().max().item()
        assert d > 1e-3, f'gelu vs {wrong} indistinguishable ({d})'
        print(f'gelu-vs-{wrong} effect on logits: max |diff| = {d:.4f}')

    # 6. final_logits_bias must matter (a head that forgets to fold it in
    # FAILS parity).
    nb = copy.deepcopy(model)
    nb.final_logits_bias.zero_()
    d = (nb(input_ids=enc_ids, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-1, f'final_logits_bias had no effect ({d})'
    print(f'final_logits_bias effect on logits: max |diff| = {d:.4f}')

    # 7. The encoder input must reach the logits through cross-attention.
    other = torch.tensor([enc_sequences[1]])
    d = (model(input_ids=other, decoder_input_ids=dec_ids).logits -
         base).abs().max().item()
    assert d > 1e-2, f'encoder ids had no effect on logits ({d})'
    print(f'encoder-input effect on logits: max |diff| = {d:.4f}')

    # 8. Decoder self-attention must be CAUSAL: changing a LATER decoder
    # token must not change EARLIER logit rows.
    dec2 = dec_ids.clone()
    dec2[0, -1] = (dec2[0, -1] + 1) % (VOCAB - 3) + 3
    l2 = model(input_ids=enc_ids, decoder_input_ids=dec2).logits
    d_early = (l2[0, :-1] - base[0, :-1]).abs().max().item()
    assert d_early < 1e-9, f'decoder attention is not causal ({d_early})'
    print('decoder causality check passed')

    # 9. Every Linear is BIASED (the BART convention) with nonzero biases,
    # so a bias-dropping import FAILS.
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m is not model.lm_head:
            assert m.bias is not None, 'unexpected bias-free Linear'
            assert m.bias.abs().max().item() > 1e-3, 'all-zero Linear bias'
    print('biased-Linear check passed')

    # 10. POST-norm placement: shifting the last block's final_layer_norm
    # bias must shift the encoder output by exactly that amount (a pre-norm
    # wiring would dilute it through the residual).
    pn = copy.deepcopy(model)
    pn.model.encoder.layers[-1].final_layer_norm.bias += 1.0
    eh = pn.model.encoder(input_ids=enc_ids).last_hidden_state
    eh_base = model.model.encoder(input_ids=enc_ids).last_hidden_state
    d = (eh - eh_base - 1.0).abs().max().item()
    assert d < 1e-9, f'encoder output is not post-normed ({d})'
    print('post-norm placement check passed')
print('all fixture self-checks passed')
