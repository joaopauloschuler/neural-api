#!/usr/bin/env python3
"""Generate a tiny RANDOM DETR object-detection parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico DetrForObjectDetection (a tiny HF ResNet backbone +
the DETR transformer encoder-decoder + class/box heads) is built from a small
config and randomly initialized (never downloaded). The reference forward is the
REAL transformers DetrForObjectDetection run in float64 (the package is
installed in this environment); the test asserts the Pascal forward matches the
HF per-query class `logits` AND the per-query `pred_boxes` tensors.

This is the repo's FIRST object-detection importer: the output modality is a
FIXED SET of object queries, each carrying a class distribution (num_labels + 1,
the last slot = "no object") and a bounding box (cxcywh, sigmoid-normalized).
INFERENCE needs no Hungarian matcher (training-only): just softmax each query's
class logits, drop the no-object slot, threshold, and read off the box.

Genuinely new code exercised here:
  - the fixed set of learned object queries (query_position_embeddings.weight),
    loaded into a TNNetLearnedPositionalEmbedding-style additive table;
  - the 2-D sinusoidal spatial position embedding over the CNN feature grid,
    computed in Pascal (matching DetrSinePositionEmbedding) and added to the
    queries+keys (but NOT values) of every encoder/decoder attention;
  - the box-regression head (3-layer ReLU MLP -> sigmoid cxcywh) and the class
    head (Linear -> num_labels + 1 logits).
The ResNet backbone is reused (conv + folded FrozenBatchNorm; HF key scheme),
as is the transformer enc-dec primitive set (q/k/v/o projections, cross-attn).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_detr_fixture.py
writes tests/fixtures/tiny_detr{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig

# ---------------- pico config ----------------
# Tiny HF ResNet backbone: 7x7-stride2 stem conv + maxpool (/4) then two
# bottleneck stages (stage0 stride 1, stage1 stride 2 => /8 total). The last
# stage width equals d_model so input_projection is a 1x1 16->16 conv.
# IMAGE 32x32 -> backbone feature grid 4x4 = 16 tokens.
IMAGE = 32
NUM_CHANNELS = 3
EMBED_SIZE = 8                     # ResNet stem width
HIDDEN_SIZES = [8, 16]            # per-stage output widths
DEPTHS = [1, 1]
D_MODEL = 16
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 2
DEC_HEADS = 2
ENC_FFN = 32
DEC_FFN = 32
NUM_QUERIES = 5
NUM_LABELS = 4                     # foreground classes (logits => +1 no-object)

torch.manual_seed(1234)
np.random.seed(1234)

bc = ResNetConfig(
    num_channels=NUM_CHANNELS,
    embedding_size=EMBED_SIZE,
    hidden_sizes=HIDDEN_SIZES,
    depths=DEPTHS,
    layer_type="bottleneck",
    hidden_act="relu",
    out_features=["stage2"],
)
cfg = DetrConfig(
    use_timm_backbone=False,
    backbone_config=bc,
    backbone=None,
    use_pretrained_backbone=False,
    num_queries=NUM_QUERIES,
    d_model=D_MODEL,
    encoder_layers=ENC_LAYERS,
    decoder_layers=DEC_LAYERS,
    encoder_attention_heads=ENC_HEADS,
    decoder_attention_heads=DEC_HEADS,
    encoder_ffn_dim=ENC_FFN,
    decoder_ffn_dim=DEC_FFN,
    num_labels=NUM_LABELS,
)
model = DetrForObjectDetection(cfg).eval()

# Re-randomize on an O(1) scale: the HF default init (std ~0.02) makes the
# transformer output near-constant (every query collapses to the same box at
# 0.5,0.5,0.5,0.5), so the parity test would pass trivially. Two separate
# scales:
#  - the BACKBONE weights are kept SMALL (std 0.08) so the ResNet's float32
#    activations stay in a tight range -> low float32-vs-float64 drift into the
#    transformer (the deep conv stack is the main error amplifier);
#  - the TRANSFORMER + head weights are LARGER (std 0.35) and the object-query
#    embedding spread is LARGE (3.0). Together these make the per-query logits
#    visibly DISTINCT (per-query std ~0.019, NOT a collapse), so the test truly
#    exercises the object-query position embedding and the decoder self/cross
#    attention wiring (a collapsed model would pass even with broken queries).
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.startswith("model.backbone"):
            std = 0.08
        else:
            std = 0.35
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=std)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=std)
    # FrozenBatchNorm running stats: keep var ~1, mean ~0 so the fold is stable.
    for name, buf in model.named_buffers():
        if name.endswith("running_var"):
            buf.fill_(1.0)
        elif name.endswith("running_mean"):
            buf.zero_()
    # Object queries: a LARGE spread so each query attends differently and the
    # per-query outputs do not collapse to a single shared detection.
    torch.nn.init.normal_(model.model.query_position_embeddings.weight,
                          mean=0.0, std=3.0)

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    x = np.random.randn(1, NUM_CHANNELS, IMAGE, IMAGE).astype(np.float64) * 0.8
    with torch.no_grad():
        out = model_d(pixel_values=torch.from_numpy(x))
    logits = out.logits.numpy()[0]      # (num_queries, num_labels + 1)
    boxes = out.pred_boxes.numpy()[0]   # (num_queries, 4) cxcywh
    img_chw = x[0]                      # (C, H, W)
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # flat (y, x, c)
    cases.append({
        "input": img_xyc.tolist(),
        "num_queries": int(logits.shape[0]),
        "num_logits": int(logits.shape[1]),       # num_labels + 1
        "logits": logits.reshape(-1).tolist(),     # flat (query, logit)
        "boxes": boxes.reshape(-1).tolist(),       # flat (query, 4) cxcywh
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_detr.safetensors"))

config = {
    "model_type": "detr",
    "num_labels": NUM_LABELS,
    "num_queries": NUM_QUERIES,
    "d_model": D_MODEL,
    "encoder_layers": ENC_LAYERS,
    "decoder_layers": DEC_LAYERS,
    "encoder_attention_heads": ENC_HEADS,
    "decoder_attention_heads": DEC_HEADS,
    "encoder_ffn_dim": ENC_FFN,
    "decoder_ffn_dim": DEC_FFN,
    "num_channels": NUM_CHANNELS,
    "image_size": IMAGE,
    "backbone_config": {
        "model_type": "resnet",
        "num_channels": NUM_CHANNELS,
        "embedding_size": EMBED_SIZE,
        "hidden_sizes": HIDDEN_SIZES,
        "depths": DEPTHS,
        "layer_type": "bottleneck",
        "hidden_act": "relu",
    },
}
with open(os.path.join(fixtures, "tiny_detr_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_detr_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_detr.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} backbone embed={EMBED_SIZE} stages={HIDDEN_SIZES} "
      f"d_model={D_MODEL} queries={NUM_QUERIES} labels={NUM_LABELS}")
print(f"  logits shape = ({cases[0]['num_queries']}, {cases[0]['num_logits']})")
print(f"  #tensors = {len(tensors)}")
sz = os.path.getsize(os.path.join(fixtures, "tiny_detr.safetensors"))
print(f"  safetensors size = {sz} bytes")
