#!/usr/bin/env python3
"""Slice a GPT-2 safetensors checkpoint into a smaller-but-real model:
keep the first KEEP_LAYERS transformer blocks and the first KEEP_VOCAB
rows of wte (row-major, so a plain byte-prefix slice). Drops the tied
lm_head.weight and the attn.bias mask buffers (the importer ignores them).
Streams tensor payloads; never holds the whole file in memory."""
import json, struct, sys

SRC, DST = sys.argv[1], sys.argv[2]
KEEP_LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
KEEP_VOCAB = int(sys.argv[4]) if len(sys.argv) > 4 else 8192

with open(SRC, 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    data_start = 8 + hlen

    def keep(name):
        if name == '__metadata__' or name == 'lm_head.weight':
            return False
        if '.attn.bias' in name or '.attn.masked_bias' in name:
            return False
        n = name[len('transformer.'):] if name.startswith('transformer.') else name
        if n.startswith('h.'):
            return int(n.split('.')[1]) < KEEP_LAYERS
        return True

    out, payload_plan, off = {}, [], 0
    for name in sorted(k for k in hdr if keep(k)):
        info = hdr[name]
        b, e = info['data_offsets']
        shape = list(info['shape'])
        base = name[len('transformer.'):] if name.startswith('transformer.') else name
        if base == 'wte.weight':  # row-major: first KEEP_VOCAB rows are a prefix
            rowbytes = (e - b) // shape[0]
            shape[0] = KEEP_VOCAB
            e = b + KEEP_VOCAB * rowbytes
        size = e - b
        out[name] = {'dtype': info['dtype'], 'shape': shape,
                     'data_offsets': [off, off + size]}
        payload_plan.append((data_start + b, size))
        off += size

    new_hdr = json.dumps(out).encode()
    with open(DST, 'wb') as g:
        g.write(struct.pack('<Q', len(new_hdr)))
        g.write(new_hdr)
        for src_off, size in payload_plan:
            f.seek(src_off)
            while size > 0:
                chunk = f.read(min(size, 1 << 20))
                g.write(chunk)
                size -= len(chunk)

print(f'wrote {DST}: {len(out)} tensors, layers={KEEP_LAYERS}, vocab={KEEP_VOCAB}')
