#!/usr/bin/env python3
"""CodeFormer parity fixture for the CodeFormer importer (BuildCodeFormer in
neural/neuralpretrained.pas, tests TestCodeFormer* in
tests/TestNeuralPretrained.pas).

Runs the OFFICIAL sczhou/CodeFormer basicsr/archs/codeformer_arch.py +
vqgan_arch.py (S-Lab License 1.0; not vendored here: pass a directory holding
the two files, or a CodeFormer checkout, with --codeformer-repo) in float64.
basicsr's registry and logger are stubbed, so basicsr itself is not needed.

The official CodeFormer.__init__ hard-codes the 512 px / nf 64 / ch_mult
1,2,2,4,4,8 net, so the pico net calls VQAutoEncoder.__init__ with tiny
arguments and then adds the transformer, the idx_pred head and the
Fuse_sft_blocks exactly as CodeFormer.__init__ does. The fuse block indices
follow the official tables: encoder after the last ResBlock of a level,
generator after the first ResBlock of a level. GroupNorm uses 4 groups
(the official 32 needs >= 32 channels); the importer reads norm_num_groups
from tiny_codeformer_config.json. get_codebook_feat gets the pico latent
shape instead of the hard-coded [b, 16, 16, 256] (and gathers in float64).
forward() is the official one.

Writes tests/fixtures/tiny_codeformer.pth ({'params_ema': state_dict}, F32),
tests/fixtures/tiny_codeformer_config.json and
tests/fixtures/tiny_codeformer_io.safetensors: input [3,32,32] in [-1,1] and
the outputs w0_adain (w=0, adain=True), w07_adain (w=0.7, adain=True, the
inference_codeformer.py path) and w07 (w=0.7, adain=False).

Usage (from the repo root):
  ~/x/bin/python -I tools/codeformer_tiny_fixture.py --codeformer-repo DIR

Coded by Claude (AI).
"""
import argparse
import importlib.util
import json
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument('--codeformer-repo', required=True,
                help='directory holding codeformer_arch.py + vqgan_arch.py '
                     '(directly or under basicsr/archs/)')
ap.add_argument('--out-dir', default='tests/fixtures')
args = ap.parse_args()

import torch
from torch import nn
from safetensors.torch import save_file

NORM_GROUPS = 4


def find_arch(name):
    for sub in ('', os.path.join('basicsr', 'archs')):
        path = os.path.join(args.codeformer_repo, sub, name)
        if os.path.isfile(path):
            return path
    sys.exit('cannot find %s under %s' % (name, args.codeformer_repo))


class _Registry:
    def register(self, obj=None):
        if obj is None:
            return lambda o: o
        return obj


def install_basicsr_stubs():
    basicsr = types.ModuleType('basicsr')
    utils = types.ModuleType('basicsr.utils')
    utils.get_root_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None)
    registry = types.ModuleType('basicsr.utils.registry')
    registry.ARCH_REGISTRY = _Registry()
    archs = types.ModuleType('basicsr.archs')
    basicsr.utils = utils
    basicsr.archs = archs
    utils.registry = registry
    for name, mod in (('basicsr', basicsr), ('basicsr.utils', utils),
                      ('basicsr.utils.registry', registry),
                      ('basicsr.archs', archs)):
        sys.modules[name] = mod


def load_module(full_name, path):
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


install_basicsr_stubs()
vq = load_module('basicsr.archs.vqgan_arch', find_arch('vqgan_arch.py'))
vq.normalize = lambda c: nn.GroupNorm(NORM_GROUPS, c, eps=1e-6, affine=True)
cf = load_module('basicsr.archs.codeformer_arch', find_arch('codeformer_arch.py'))

IMG, NF, CH_MULT, RES_BLOCKS = 32, 8, [1, 2, 2, 1], 2
ATTN_RES = [IMG >> (len(CH_MULT) - 1)]
CODEBOOK, EMB_DIM = 32, 16
DIM_EMBD, N_HEAD, N_LAYERS = 32, 2, 2
LATENT = IMG >> (len(CH_MULT) - 1)
CONNECT = ['8', '16', '32']


def fuse_tables():
    """Block indices in the official tables' sense, for the pico layout."""
    levels = len(CH_MULT)
    enc, pos, res = {}, 1, IMG
    for i in range(levels):
        for r in range(RES_BLOCKS):
            if r == RES_BLOCKS - 1:
                enc[str(res)] = pos
            pos += 1
            if res in ATTN_RES:
                pos += 1
        if i != levels - 1:
            pos += 1
            res //= 2
    gen, pos, res = {}, 4, LATENT
    for i in reversed(range(levels)):
        for r in range(RES_BLOCKS):
            if r == 0:
                gen[str(res)] = pos
            pos += 1
            if res in ATTN_RES:
                pos += 1
        if i != 0:
            pos += 1
            res *= 2
    return enc, gen


def build_pico():
    m = cf.CodeFormer.__new__(cf.CodeFormer)
    vq.VQAutoEncoder.__init__(m, IMG, NF, CH_MULT, 'nearest', RES_BLOCKS,
                              ATTN_RES, CODEBOOK, EMB_DIM)
    m.connect_list = CONNECT
    m.n_layers = N_LAYERS
    m.dim_embd = DIM_EMBD
    m.dim_mlp = DIM_EMBD * 2
    m.position_emb = nn.Parameter(torch.zeros(LATENT * LATENT, DIM_EMBD))
    m.feat_emb = nn.Linear(EMB_DIM, DIM_EMBD)
    m.ft_layers = nn.Sequential(*[cf.TransformerSALayer(
        embed_dim=DIM_EMBD, nhead=N_HEAD, dim_mlp=m.dim_mlp, dropout=0.0)
        for _ in range(N_LAYERS)])
    m.idx_pred_layer = nn.Sequential(nn.LayerNorm(DIM_EMBD),
                                     nn.Linear(DIM_EMBD, CODEBOOK, bias=False))
    m.channels = {str(IMG >> i): NF * CH_MULT[i] for i in range(len(CH_MULT))}
    m.fuse_encoder_block, m.fuse_generator_block = fuse_tables()
    m.fuse_convs_dict = nn.ModuleDict()
    for f_size in CONNECT:
        in_ch = m.channels[f_size]
        m.fuse_convs_dict[f_size] = cf.Fuse_sft_block(in_ch, in_ch)
    # The official one-hot matmul casts to float32; this row gather is the
    # same lookup in the codebook's dtype (float64 here).
    m.quantize.get_codebook_feat = lambda idx, shape: \
        m.quantize.embedding.weight[idx.view(-1)].view(
            shape[0], LATENT, LATENT, EMB_DIM).permute(0, 3, 1, 2).contiguous()
    return m


torch.manual_seed(7)
net = build_pico()
with torch.no_grad():
    for name, p in net.named_parameters():
        if name.endswith('norm1.weight') or name.endswith('norm2.weight') or \
           name.endswith('norm.weight'):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        elif name.endswith('bias'):
            p.copy_(0.1 * torch.randn_like(p))
        elif name == 'quantize.embedding.weight':
            p.copy_(torch.randn_like(p))
        elif name == 'position_emb':
            p.copy_(0.5 * torch.randn_like(p))
        elif p.dim() > 1:
            fan_in = p[0].numel()
            p.copy_(torch.randn_like(p) / fan_in ** 0.5)
        else:
            p.copy_(1 + 0.2 * torch.randn_like(p))
    # Spread the code logits so float32 cannot flip a top-1 pick.
    net.idx_pred_layer[1].weight.mul_(4.0)
net = net.double().eval()

x = torch.rand(1, 3, IMG, IMG, generator=torch.Generator().manual_seed(11),
               dtype=torch.float64) * 2 - 1
io = {'input': x[0].float().contiguous()}
with torch.no_grad():
    for key, w, adain in (('w0_adain', 0.0, True), ('w07_adain', 0.7, True),
                          ('w07', 0.7, False)):
        out, logits, lq = net(x, w=w, adain=adain)
        io[key] = out[0].float().contiguous()
        top2 = torch.topk(logits[0], 2, dim=1).values
        print(key, 'out range', out.min().item(), out.max().item(),
              'min top-1 margin', (top2[:, 0] - top2[:, 1]).min().item())
    io['codes'] = torch.topk(logits[0], 1, dim=1).indices[:, 0].float()

os.makedirs(args.out_dir, exist_ok=True)
state = {k: v.float().contiguous() for k, v in net.state_dict().items()}
torch.save({'params_ema': state}, os.path.join(args.out_dir, 'tiny_codeformer.pth'))
save_file(io, os.path.join(args.out_dir, 'tiny_codeformer_io.safetensors'))
with open(os.path.join(args.out_dir, 'tiny_codeformer_config.json'), 'w') as f:
    json.dump({'n_head': N_HEAD, 'norm_num_groups': NORM_GROUPS}, f)
    f.write('\n')
print('fuse tables', net.fuse_encoder_block, net.fuse_generator_block)
print('params', sum(p.numel() for p in net.parameters()))
