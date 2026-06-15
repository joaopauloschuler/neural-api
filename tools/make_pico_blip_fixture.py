#!/usr/bin/env python3
"""Generate a tiny RANDOM BLIP image-captioning parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico BlipForConditionalGeneration (a tiny ViT image
encoder + a BERT-style text DECODER that cross-attends to the image features +
an LM head) is built from a small config and randomly initialized (never
downloaded). The reference forward is the REAL transformers
BlipForConditionalGeneration run in float64 (the package is installed in this
environment); the test asserts the Pascal forward matches the per-position
next-token `logits` over the vocabulary, and that the greedy caption ids match.

This is the repo's FIRST GENERATIVE vision-language importer of the
ENCODER-DECODER kind: a ViT image encoder feeds a BERT-style causal text
decoder through CROSS-ATTENTION (TNNetCrossAttention), which autoregressively
generates a caption. It reuses the two-net (encoder + cross-attending decoder)
convention of the T5/Marian/Pegasus importers and the DecodeSeq2Seq* helpers.

The vision tower is the CLIP/ViT primitive set (patch conv + class token +
learned positions + pre-LN encoder blocks + post_layernorm). The text decoder
is the BERT post-LN block (causal self-attention + cross-attention + GELU FFN)
plus the BERT LM prediction head (transform dense+GELU+LN, then the vocab
decoder).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_blip_fixture.py
writes tests/fixtures/tiny_blip{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import BlipForConditionalGeneration, BlipConfig
from transformers.models.blip.configuration_blip import (
    BlipTextConfig, BlipVisionConfig)

# ---------------- pico config ----------------
IMAGE = 16
PATCH = 8                 # 2x2 = 4 patches (+1 class token = 5 vision tokens)
NUM_CHANNELS = 3
HIDDEN = 24
INTER = 48
VIS_LAYERS = 2
VIS_HEADS = 3
TXT_LAYERS = 2
TXT_HEADS = 3
VOCAB = 40
MAX_POS = 32
BOS = 1
EOS = 2
PAD = 0

torch.manual_seed(1234)
np.random.seed(1234)

vis = BlipVisionConfig(
    hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=VIS_LAYERS, num_attention_heads=VIS_HEADS,
    image_size=IMAGE, patch_size=PATCH, num_channels=NUM_CHANNELS,
    hidden_act="gelu", layer_norm_eps=1e-5)
txt = BlipTextConfig(
    vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=TXT_LAYERS, num_attention_heads=TXT_HEADS,
    max_position_embeddings=MAX_POS, hidden_act="gelu",
    layer_norm_eps=1e-12, is_decoder=True,
    bos_token_id=BOS, eos_token_id=EOS, sep_token_id=EOS, pad_token_id=PAD)
cfg = BlipConfig(text_config=txt.to_dict(), vision_config=vis.to_dict(),
                 projection_dim=HIDDEN, image_text_hidden_size=HIDDEN)
cfg.text_config.bos_token_id = BOS
cfg.text_config.eos_token_id = EOS
cfg.text_config.sep_token_id = EOS
cfg.text_config.pad_token_id = PAD

model = BlipForConditionalGeneration(cfg).eval()

# Re-randomize on an O(1) scale: HF default init (std ~0.02) makes the decoder
# output near-constant over positions, so a parity test could pass with broken
# wiring. Larger weights make the per-position logits visibly DISTINCT and
# really exercise the cross-attention to the image features.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "LayerNorm" in name or "layer_norm" in name or name.endswith(
                "norm.weight") or "post_layernorm" in name:
            if name.endswith(".weight"):
                torch.nn.init.normal_(p, mean=1.0, std=0.05)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.05)
        elif "embeddings" in name and "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.20)
        elif "class_embedding" in name or "position_embedding" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.20)
        elif "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.18)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.10)

# ---------------- float64 reference forward ----------------
model_d = model.double()

# One tiny image, fixed pixel values.
x = (np.random.randn(1, NUM_CHANNELS, IMAGE, IMAGE).astype(np.float64) * 0.8)
img_chw = x[0]
img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # flat (y, x, c)

# A short input token prefix the decoder conditions on; the reference target is
# the NEXT-token logits over the vocab at EVERY prefix position.
prefix = [BOS, 7, 19, 3]
input_ids = torch.tensor([prefix], dtype=torch.long)

with torch.no_grad():
    pix = torch.from_numpy(x)
    out = model_d(pixel_values=pix, input_ids=input_ids)
    logits = out.logits.numpy()[0]      # (seqlen, vocab)

# Greedy caption from the model.generate path (ids only).
with torch.no_grad():
    gen = model.double().generate(
        pixel_values=pix, max_length=12, num_beams=1, do_sample=False)
    caption_ids = gen.numpy()[0].tolist()

cases = [{
    "input": img_xyc.tolist(),
    "prefix": prefix,
    "seq_len": int(logits.shape[0]),
    "vocab": int(logits.shape[1]),
    "logits": logits.reshape(-1).tolist(),  # flat (pos, vocab)
    "caption_ids": [int(t) for t in caption_ids],
}]

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked"):
        continue
    arr = v.detach().cpu().numpy().astype(np.float32)
    tensors[k] = np.ascontiguousarray(arr)
save_file(tensors, os.path.join(fixtures, "tiny_blip.safetensors"))

config = {
    "model_type": "blip",
    "text_config": {
        "vocab_size": VOCAB, "hidden_size": HIDDEN,
        "intermediate_size": INTER, "num_hidden_layers": TXT_LAYERS,
        "num_attention_heads": TXT_HEADS,
        "max_position_embeddings": MAX_POS, "hidden_act": "gelu",
        "layer_norm_eps": 1e-12, "bos_token_id": BOS, "eos_token_id": EOS,
        "sep_token_id": EOS, "pad_token_id": PAD,
    },
    "vision_config": {
        "hidden_size": HIDDEN, "intermediate_size": INTER,
        "num_hidden_layers": VIS_LAYERS, "num_attention_heads": VIS_HEADS,
        "image_size": IMAGE, "patch_size": PATCH,
        "num_channels": NUM_CHANNELS, "hidden_act": "gelu",
        "layer_norm_eps": 1e-5,
    },
}
with open(os.path.join(fixtures, "tiny_blip_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_blip_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_blip.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} patch={PATCH} hidden={HIDDEN} "
      f"vis_layers={VIS_LAYERS} txt_layers={TXT_LAYERS} vocab={VOCAB}")
print(f"  logits shape = ({cases[0]['seq_len']}, {cases[0]['vocab']})")
print(f"  caption_ids = {cases[0]['caption_ids']}")
print(f"  #tensors = {len(tensors)}")
sz = os.path.getsize(os.path.join(fixtures, "tiny_blip.safetensors"))
print(f"  safetensors size = {sz} bytes")
