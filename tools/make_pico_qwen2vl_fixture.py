#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen2-VL parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

Qwen2-VL = a native-dynamic-resolution ViT (Conv3d patch embed + window
attention + a 2x2 spatial patch merger) producing MERGED visual tokens that
are SPLICED into a Qwen2 decoder's token-embedding sequence at the
image_token_id placeholders. The GENUINELY NEW behaviour versus LLaVA
(causal-everywhere) and PaliGemma (prefix-LM bidirectional) is M-RoPE
(Multimodal Rotary Position Embedding): each token's rotary INDEX is a 3-D
(temporal, height, width) grid position rather than the scalar sequence index.
Vision tokens carry the patch grid's (t, h, w); text tokens carry the same
scalar in all three sections (1-D RoPE). The head_dim/2 channel-pairs are
split into three contiguous mrope_section chunks.

v1 scope (this fixture): a SINGLE still image (T=1) + text. The Pascal side
takes the MERGED visual tokens as input (precomputed here, exactly the HF
get_image_features output) and tests the M-RoPE 3-D index path end to end on
the decoder's next-token logits. The full Conv3d vision tower is a follow-up.

The image grid is 1x4x4 -> 16 patches -> 4 merged tokens (a 2x2 merged grid)
so the M-RoPE height/width positions actually DIFFER across the image tokens
(a 1-token image would collapse M-RoPE to a scalar and not exercise it).

The committed fixture (KB-scale) in tests/fixtures/:
  tiny_qwen2vl.safetensors  - the Qwen2 DECODER state_dict under the standard
      model.embed_tokens / model.layers.N.* / model.norm / lm_head names
      (extracted from the Qwen2-VL model.language_model.* keys), so the
      existing BuildQwen2/Llama importer reads it directly.
  tiny_qwen2vl_config.json  - a FLAT qwen2 text config (the Llama/Qwen2 reader
      handles it) PLUS the extra Qwen2-VL keys image_token_id,
      spatial_merge_size, mrope_section.
  tiny_qwen2vl_logits.json  - the float64 oracle: the pinned token ids
      (image_token_id at the 4 merged-image slots), the merged image embeds
      [4][hidden], the merged grid (h, w), the 3-D position ids [3][seq], and
      the next-token logits [seq][vocab] for the mixed image+text prompt.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/make_pico_qwen2vl_fixture.py
Needs torch + transformers + safetensors.
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

HID = 32            # text hidden_size
INTER = 64          # text intermediate_size (SwiGLU width)
N_LAYER = 2
N_HEAD = 4
N_KV = 2            # GQA
HEAD_DIM = HID // N_HEAD          # 8 -> head_dim/2 = 4
MROPE_SECTION = [1, 1, 2]         # sums to head_dim/2 = 4
VOCAB = 40
MAX_POS = 128
IMAGE_TOKEN = 39

# Vision grid: 1 x 4 x 4 = 16 patches -> /spatial_merge_size^2 (=4) = 4 merged
# tokens forming a 2 x 2 MERGED grid.
GRID_T, GRID_H, GRID_W = 1, 4, 4
SPATIAL_MERGE = 2
MERGED_H = GRID_H // SPATIAL_MERGE   # 2
MERGED_W = GRID_W // SPATIAL_MERGE   # 2
N_MERGED = MERGED_H * MERGED_W       # 4

torch.manual_seed(20260615)

text_config = dict(
    model_type="qwen2", hidden_size=HID, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    num_key_value_heads=N_KV, vocab_size=VOCAB,
    max_position_embeddings=MAX_POS, rms_norm_eps=1e-6, rope_theta=10000.0,
    rope_scaling={"type": "mrope", "mrope_section": MROPE_SECTION},
    tie_word_embeddings=False)
vision_config = dict(
    depth=2, embed_dim=16, hidden_size=HID, num_heads=2, in_chans=3,
    patch_size=14, spatial_merge_size=SPATIAL_MERGE, temporal_patch_size=2)
cfg = Qwen2VLConfig(text_config=text_config, vision_config=vision_config,
                    image_token_id=IMAGE_TOKEN)

model = Qwen2VLForConditionalGeneration(cfg).double().eval()

# --- pinned image + prompt ---
n_patch = GRID_T * GRID_H * GRID_W
pv = torch.randn(n_patch, 3 * 2 * 14 * 14, dtype=torch.float64)
grid_thw = torch.tensor([[GRID_T, GRID_H, GRID_W]])

# prompt: text prefix, then N_MERGED image tokens, then text suffix
ids = [5, 6, 7] + [IMAGE_TOKEN] * N_MERGED + [8, 9, 10]
input_ids = torch.tensor([ids])
mm = torch.tensor([[0, 0, 0] + [1] * N_MERGED + [0, 0, 0]])
seq = len(ids)

with torch.no_grad():
    out = model(input_ids=input_ids, pixel_values=pv,
                image_grid_thw=grid_thw, mm_token_type_ids=mm)
    logits = out.logits[0]  # (seq, vocab)
    feats = model.model.get_image_features(pixel_values=pv,
                                           image_grid_thw=grid_thw)
    merged = feats.pooler_output
    merged = merged[0] if isinstance(merged, (list, tuple)) else merged
    pos_ids, _ = model.model.get_rope_index(input_ids, mm,
                                            image_grid_thw=grid_thw)
    pos_ids = pos_ids[:, 0, :]  # (3, seq)

# --- self-check: the merged image embeds line up with the spliced positions
assert merged.shape == (N_MERGED, HID), merged.shape
assert logits.shape == (seq, VOCAB), logits.shape

# --- decoder state_dict under standard model.* / lm_head names ---
sd = model.state_dict()
out_sd = {}
PREFIX = "model.language_model."
for k, v in sd.items():
    if k.startswith(PREFIX):
        out_sd["model." + k[len(PREFIX):]] = v.contiguous().to(torch.float32)
    elif k == "lm_head.weight":
        out_sd["lm_head.weight"] = v.contiguous().to(torch.float32)
# sanity: the standard decoder keys must be present
assert "model.embed_tokens.weight" in out_sd
assert "model.layers.0.self_attn.q_proj.bias" in out_sd  # Qwen2 QKV bias
assert "lm_head.weight" in out_sd

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures")

save_file(out_sd, os.path.join(FIX, "tiny_qwen2vl.safetensors"))

# flat qwen2 config + extra Qwen2-VL keys
flat_cfg = dict(
    model_type="qwen2", hidden_size=HID, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    num_key_value_heads=N_KV, vocab_size=VOCAB,
    max_position_embeddings=MAX_POS, rms_norm_eps=1e-6, rope_theta=10000.0,
    tie_word_embeddings=False,
    # extra Qwen2-VL keys read by ReadQwen2VLConfigFromJSONFile:
    image_token_id=IMAGE_TOKEN, spatial_merge_size=SPATIAL_MERGE,
    mrope_section=MROPE_SECTION)
with open(os.path.join(FIX, "tiny_qwen2vl_config.json"), "w") as f:
    json.dump(flat_cfg, f, indent=2)

oracle = dict(
    token_ids=ids,
    image_token_id=IMAGE_TOKEN,
    merged_h=MERGED_H, merged_w=MERGED_W,
    image_embeds=[[float(x) for x in row] for row in merged.tolist()],
    position_ids=[[int(x) for x in row] for row in pos_ids.tolist()],
    logits=[[float(x) for x in row] for row in logits.tolist()],
)
with open(os.path.join(FIX, "tiny_qwen2vl_logits.json"), "w") as f:
    json.dump(oracle, f)

print("wrote tiny_qwen2vl.safetensors / _config.json / _logits.json")
print("seq", seq, "merged grid", MERGED_H, "x", MERGED_W,
      "= ", N_MERGED, "image tokens")
print("position_ids (3, seq):")
for r in pos_ids.tolist():
    print("  ", r)
print("last-token logits[:6]", [round(x, 5) for x in logits[-1, :6].tolist()])
