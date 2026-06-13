#!/usr/bin/env python3
"""Splits tests/fixtures/tiny_llama.safetensors into a 2-shard HF-style
sharded checkpoint for the TNNetSafeTensorsReader sharded-load test:

  tests/fixtures/tiny_llama-00001-of-00002.safetensors
  tests/fixtures/tiny_llama-00002-of-00002.safetensors
  tests/fixtures/tiny_llama.safetensors.index.json

The split mirrors how HF shards real checkpoints (whole tensors assigned to
shards; here: embed_tokens + layer 0 in shard 1, the rest in shard 2). The
index.json carries the standard {"metadata": {"total_size": N},
"weight_map": {tensor: shard}} layout. Run from the repo root:

  python3 tools/shard_tiny_llama_fixture.py
"""
import json
import os

from safetensors import safe_open
from safetensors.numpy import save_file

FIXDIR = os.path.join("tests", "fixtures")
SRC = os.path.join(FIXDIR, "tiny_llama.safetensors")
SHARD1 = "tiny_llama-00001-of-00002.safetensors"
SHARD2 = "tiny_llama-00002-of-00002.safetensors"
INDEX = os.path.join(FIXDIR, "tiny_llama.safetensors.index.json")


def shard_of(name):
    if name == "model.embed_tokens.weight" or name.startswith("model.layers.0."):
        return SHARD1
    return SHARD2


def main():
    tensors = {}
    with safe_open(SRC, framework="numpy") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    shards = {SHARD1: {}, SHARD2: {}}
    weight_map = {}
    for name, value in sorted(tensors.items()):
        shard = shard_of(name)
        shards[shard][name] = value
        weight_map[name] = shard
    assert shards[SHARD1] and shards[SHARD2], "both shards must be non-empty"

    total_size = 0
    for shard, content in shards.items():
        save_file(content, os.path.join(FIXDIR, shard))
        total_size += sum(v.nbytes for v in content.values())

    with open(INDEX, "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            f,
            indent=1,
            sort_keys=True,
        )
        f.write("\n")
    print(f"wrote {SHARD1} ({len(shards[SHARD1])} tensors), "
          f"{SHARD2} ({len(shards[SHARD2])} tensors), {INDEX}")


if __name__ == "__main__":
    main()
