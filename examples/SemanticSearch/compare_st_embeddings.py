#!/usr/bin/env python3
"""Parity check for examples/SemanticSearch: compares the Pascal sentence
embeddings (SemanticSearch <modeldir> -dump pas_embeddings.json) against the
HuggingFace reference for the same checkpoint.

Reference: sentence_transformers model.encode() when the package is
installed, else plain transformers AutoModel mean-pooled last_hidden_state +
L2 normalize -- for all-MiniLM-L6-v2 the two are the same computation
(its pooling config is attention-mask-aware mean pooling + normalize).

Usage:
  python3 compare_st_embeddings.py <modeldir> <pas_embeddings.json>

Passes when cosine(Pascal, reference) > 0.999 for every sentence.
"""
import json
import sys

import torch


def reference_embeddings(modeldir, sentences):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(modeldir)
        return torch.tensor(model.encode(sentences, normalize_embeddings=True))
    except ImportError:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(modeldir)
        model = AutoModel.from_pretrained(modeldir).eval()
        out = []
        for sent in sentences:
            enc = tok(sent, return_tensors='pt')
            with torch.no_grad():
                h = model(**enc).last_hidden_state[0]
            # single unpadded sentence: attention mask is all ones, so
            # mask-aware mean pooling reduces to a plain mean
            ref = h.mean(0)
            out.append(ref / ref.norm())
        return torch.stack(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    modeldir, dump_path = sys.argv[1], sys.argv[2]
    dump = json.load(open(dump_path))
    sentences = dump['sentences']
    pascal = torch.tensor(dump['embeddings'], dtype=torch.float32)
    refs = reference_embeddings(modeldir, sentences)
    worst = 1.0
    for sent, pas, ref in zip(sentences, pascal, refs):
        cos = float(torch.dot(ref, pas) / (pas.norm() * ref.norm()))
        worst = min(worst, cos)
        status = 'OK ' if cos > 0.999 else 'FAIL'
        print(f"{status} cosine={cos:.6f}  {sent[:60]}")
    print(f"worst cosine: {worst:.6f}")
    sys.exit(0 if worst > 0.999 else 1)


if __name__ == '__main__':
    main()
