#!/usr/bin/env python3
"""Slice a Llama-architecture safetensors checkpoint into a
smaller-but-real model: keep the first KEEP_LAYERS decoder blocks and the
first KEEP_VOCAB rows of embed_tokens (and of the untied lm_head). Rows are
row-major, so the vocab cut is a plain byte-prefix slice. Drops any
serialized rotary_emb.inv_freq buffers (the importer ignores them).
Streams tensor payloads; never holds the whole file in memory.

Coded by Claude (AI).

Usage:
  python3 slice_llama.py <src_dir_or_safetensors> <dst_dir> [layers] [vocab]

src may be a directory containing model.safetensors + config.json or a
.safetensors path (config.json is then looked up next to it). dst_dir gets
model.safetensors + a rewritten config.json.
"""
import json, os, struct, sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, DST_DIR = sys.argv[1], sys.argv[2]
KEEP_LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
KEEP_VOCAB = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

if os.path.isdir(SRC):
    src_st = os.path.join(SRC, 'model.safetensors')
    src_cfg = os.path.join(SRC, 'config.json')
else:
    src_st = SRC
    src_cfg = os.path.join(os.path.dirname(SRC) or '.', 'config.json')

os.makedirs(DST_DIR, exist_ok=True)
cfg = json.load(open(src_cfg))

with open(src_st, 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    data_start = 8 + hlen

    def layer_of(name):
        # model.layers.N.* or layers.N.*
        parts = name.split('.')
        if 'layers' in parts:
            return int(parts[parts.index('layers') + 1])
        return None

    def keep(name):
        if name == '__metadata__' or 'rotary_emb.inv_freq' in name:
            return False
        ln = layer_of(name)
        return ln is None or ln < KEEP_LAYERS

    out, payload_plan, off = {}, [], 0
    for name in sorted(k for k in hdr if keep(k)):
        info = hdr[name]
        b, e = info['data_offsets']
        shape = list(info['shape'])
        if name.endswith(('embed_tokens.weight', 'lm_head.weight')):
            # row-major [vocab, d]: the first KEEP_VOCAB rows are a prefix
            rowbytes = (e - b) // shape[0]
            shape[0] = min(KEEP_VOCAB, shape[0])
            e = b + shape[0] * rowbytes
        size = e - b
        out[name] = {'dtype': info['dtype'], 'shape': shape,
                     'data_offsets': [off, off + size]}
        payload_plan.append((data_start + b, size))
        off += size

    new_hdr = json.dumps(out).encode()
    dst_st = os.path.join(DST_DIR, 'model.safetensors')
    with open(dst_st, 'wb') as g:
        g.write(struct.pack('<Q', len(new_hdr)))
        g.write(new_hdr)
        for src_off, size in payload_plan:
            f.seek(src_off)
            while size > 0:
                chunk = f.read(min(size, 1 << 20))
                g.write(chunk)
                size -= len(chunk)

cfg['num_hidden_layers'] = KEEP_LAYERS
cfg['vocab_size'] = min(KEEP_VOCAB, cfg['vocab_size'])
with open(os.path.join(DST_DIR, 'config.json'), 'w') as g:
    json.dump(cfg, g, indent=1)

print(f'wrote {dst_st}: {len(out)} tensors, layers={KEEP_LAYERS}, '
      f'vocab={cfg["vocab_size"]}')
