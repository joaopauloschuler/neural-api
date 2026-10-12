#!/usr/bin/env python3
"""Kokoro-82M (hexgrad/Kokoro-82M, StyleTTS2 + iSTFTNet) parity references
for the real-checkpoint port (tasklist "Kokoro-82M TTS, real checkpoint",
tasks 0.1 and 0.2).

Two modes, both running the `kokoro` 0.9.4 package's own torch modules with
ZERO noise in SineGen / SourceModuleHnNSF (the Pascal default):

  real  Loads the cached HF snapshot OFFLINE, phonemizes one sentence with
        misaki, and dumps every stage of the forward graph. Large (tens of MB),
        so it goes OUTSIDE the repo (default ~/.cache/neural-api/kokoro82m_ref/):
          kokoro82m_ref.safetensors  every stage tensor, float32
          kokoro82m_ref.json         text, phonemes, input_ids, pred_dur,
                                     stage names in graph order + shapes
          kokoro82m_ref.wav          the final audio, 24 kHz int16 mono

  tiny  Builds tiny-width versions of the same package classes with random
        weights (fixed seed) and writes them in the REAL on-disk formats, as a
        miniature snapshot folder, plus the same per-stage reference:
          tests/fixtures/tiny_kokoro82m/config.json         real schema
          tests/fixtures/tiny_kokoro82m/kokoro-v1_0.pth     torch.save, dict of 5
              state dicts, keys prefixed `module.`, legacy weight_g/weight_v,
              no AdaIN InstanceNorm affine keys (as in the real file)
          tests/fixtures/tiny_kokoro82m/voices/af_heart.pt  BARE tensor [R,1,2*style]
          tests/fixtures/tiny_kokoro82m_ref.safetensors     stage tensors, float32
          tests/fixtures/tiny_kokoro82m_ref.json            metadata as above

Deviations from the package code (both deliberate):
  - The package's Decoder hard-codes its widths (encode/decode 1024, asr_res
    64, last decode block 512). TinyDecoder below is a line-for-line copy with
    only those constants made parameters. The real checkpoint config does not
    carry them; a loader must read them from the weight shapes.
  - KModel.forward_with_tokens splits ref_s at a literal 128; staged_forward
    splits at style_dim (= 128 in the real config), so the tiny model can use a
    small style_dim. The real mode checks staged_forward's audio against
    KModel.forward_with_tokens bit for bit.

Stage names (graph order, batch dim kept, torch layouts [B,C,T] / [B,T,C]):
  input_ids, ref_s, bert_embeddings, bert_hidden_in, bert_layer{0..L-1}, bert,
  d_en, dur_enc_adanorm{0..n-1}, d, duration_lstm, duration_logits, duration,
  pred_dur, en, f0n_shared, F0_block{0..2}, F0_pred, N_block{0..2}, N_pred,
  text_embedding, text_cnn{0..n-1}, t_en, asr,
  dec_F0_conv, dec_N_conv, dec_encode, dec_asr_res, dec_decode{0..3},
  gen_f0_upsampled, gen_har_source, gen_uv, gen_har_spec, gen_har_phase,
  per upsample stage i: gen_stage{i}_leaky, gen_stage{i}_noise_conv,
  gen_stage{i}_noise_res, gen_stage{i}_ups, gen_stage{i}_pad (last stage only),
  gen_stage{i}_sum, gen_stage{i}_resblock{j}, gen_stage{i}_mrf;
  gen_post_leaky, gen_conv_post, gen_spec, gen_phase, audio.
input_ids and pred_dur are also stored as float32 tensors; the JSON holds them
as exact integers.

Coded by Claude (AI).

Usage (from the repo root; both under the 3 GB cap):
  ( ulimit -v 3145728; HF_HUB_OFFLINE=1 /home/bpsa/x/bin/python -I \
      tools/make_kokoro82m_fixture.py real [--out DIR] [--snapshot DIR] )
  ( ulimit -v 3145728; /home/bpsa/x/bin/python -I \
      tools/make_kokoro82m_fixture.py tiny )
"""
import argparse
import collections
import json
import math
import os
import sys
import wave

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AlbertConfig

import kokoro
import kokoro.istftnet as istftnet
from kokoro.istftnet import AdainResBlk1d, Generator
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.normpath(os.path.join(HERE, "..", "tests", "fixtures"))
DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/"
    "f3ff3571791e39611d31c381e3a41a3af07b4987")
DEFAULT_REAL_OUT = os.path.expanduser("~/.cache/neural-api/kokoro82m_ref")
REAL_TEXT = "Hello world. This is a test."
VOICE = "af_heart"
SAMPLE_RATE = 24000
PART_NAMES = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


class _ZeroNoiseTorch:
    """Stands in for `torch` inside kokoro.istftnet: rand / randn_like return zeros."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def rand(*args, **kwargs):
        return torch.zeros(*args, **kwargs)

    @staticmethod
    def randn_like(tensor, *args, **kwargs):
        return torch.zeros_like(tensor, *args, **kwargs)


istftnet.torch = _ZeroNoiseTorch()


class TinyDecoder(nn.Module):
    """kokoro.istftnet.Decoder with its hard-coded widths (1024, 64, 512) as
    parameters; forward is the package's, line for line."""

    def __init__(self, dim_in, style_dim, dim_out,
                 resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                 resblock_dilation_sizes, upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 decoder_width=1024, asr_res_width=64, disable_complex=False):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, decoder_width, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(decoder_width + 2 + asr_res_width, decoder_width, style_dim))
        self.decode.append(AdainResBlk1d(decoder_width + 2 + asr_res_width, decoder_width, style_dim))
        self.decode.append(AdainResBlk1d(decoder_width + 2 + asr_res_width, decoder_width, style_dim))
        self.decode.append(AdainResBlk1d(decoder_width + 2 + asr_res_width, upsample_initial_channel,
                                         style_dim, upsample=True))
        self.F0_conv = nn.utils.weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = nn.utils.weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(dim_in, asr_res_width, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates,
                                   upsample_initial_channel, resblock_dilation_sizes,
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size,
                                   disable_complex=disable_complex)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, F0_curve)
        return x


class StageRecorder:
    """Ordered name -> float32 tensor store; hooks append numbered stages."""

    def __init__(self):
        self.stages = collections.OrderedDict()

    def put(self, name, tensor):
        assert name not in self.stages, name
        self.stages[name] = tensor.detach().to(torch.float32).contiguous().clone()

    def hook(self, prefix, numbered=True):
        def record(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if not numbered:
                self.put(prefix, output)
                return
            index = sum(1 for k in self.stages if k.startswith(prefix) and k[len(prefix):].isdigit())
            self.put(prefix + str(index), output)
        return record


def staged_forward(parts, input_ids, ref_s, style_dim, rec, speed=1.0):
    """KModel.forward_with_tokens + Decoder/Generator/F0Ntrain forwards inlined,
    calling the package modules and recording each stage; returns (audio, pred_dur)."""
    bert, bert_encoder, predictor, text_encoder, decoder = (parts[n] for n in PART_NAMES)
    handles = [
        bert.embeddings.register_forward_hook(rec.hook("bert_embeddings", False)),
        bert.encoder.embedding_hidden_mapping_in.register_forward_hook(rec.hook("bert_hidden_in", False)),
        bert.encoder.albert_layer_groups[0].albert_layers[0].register_forward_hook(rec.hook("bert_layer")),
        text_encoder.embedding.register_forward_hook(rec.hook("text_embedding", False)),
    ]
    for block in predictor.text_encoder.lstms:
        if not isinstance(block, nn.LSTM):
            handles.append(block.register_forward_hook(rec.hook("dur_enc_adanorm")))
    for conv_block in text_encoder.cnn:
        handles.append(conv_block.register_forward_hook(rec.hook("text_cnn")))
    try:
        rec.put("input_ids", input_ids)
        rec.put("ref_s", ref_s)
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = bert(input_ids, attention_mask=(~text_mask).int())
        rec.put("bert", bert_dur)
        d_en = bert_encoder(bert_dur).transpose(-1, -2)
        rec.put("d_en", d_en)
        s = ref_s[:, style_dim:]
        d = predictor.text_encoder(d_en, s, input_lengths, text_mask)
        rec.put("d", d)
        x, _ = predictor.lstm(d)
        rec.put("duration_lstm", x)
        duration = predictor.duration_proj(x)
        rec.put("duration_logits", duration)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        rec.put("duration", duration)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        rec.put("pred_dur", pred_dur)
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        rec.put("en", en)
        # ProsodyPredictor.F0Ntrain
        x, _ = predictor.shared(en.transpose(-1, -2))
        rec.put("f0n_shared", x)
        F0 = x.transpose(-1, -2)
        for index, block in enumerate(predictor.F0):
            F0 = block(F0, s)
            rec.put(f"F0_block{index}", F0)
        F0_pred = predictor.F0_proj(F0).squeeze(1)
        rec.put("F0_pred", F0_pred)
        N = x.transpose(-1, -2)
        for index, block in enumerate(predictor.N):
            N = block(N, s)
            rec.put(f"N_block{index}", N)
        N_pred = predictor.N_proj(N).squeeze(1)
        rec.put("N_pred", N_pred)
        t_en = text_encoder(input_ids, input_lengths, text_mask)
        rec.put("t_en", t_en)
        asr = t_en @ pred_aln_trg
        rec.put("asr", asr)
        audio = decoder_forward(decoder, asr, F0_pred, N_pred, ref_s[:, :style_dim], rec).squeeze()
        rec.put("audio", audio)
    finally:
        for handle in handles:
            handle.remove()
    return audio, pred_dur


def decoder_forward(decoder, asr, F0_curve, N, s, rec):
    """Decoder.forward, recording each stage."""
    F0 = decoder.F0_conv(F0_curve.unsqueeze(1))
    rec.put("dec_F0_conv", F0)
    N = decoder.N_conv(N.unsqueeze(1))
    rec.put("dec_N_conv", N)
    x = torch.cat([asr, F0, N], axis=1)
    x = decoder.encode(x, s)
    rec.put("dec_encode", x)
    asr_res = decoder.asr_res(asr)
    rec.put("dec_asr_res", asr_res)
    res = True
    for index, block in enumerate(decoder.decode):
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        rec.put(f"dec_decode{index}", x)
        if block.upsample_type != "none":
            res = False
    return generator_forward(decoder.generator, x, s, F0_curve, rec)


def generator_forward(gen, x, s, f0, rec):
    """Generator.forward, recording each stage."""
    f0 = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
    rec.put("gen_f0_upsampled", f0)
    har_source, _noi_source, uv = gen.m_source(f0)
    har_source = har_source.transpose(1, 2).squeeze(1)
    rec.put("gen_har_source", har_source)
    rec.put("gen_uv", uv)
    har_spec, har_phase = gen.stft.transform(har_source)
    rec.put("gen_har_spec", har_spec)
    rec.put("gen_har_phase", har_phase)
    har = torch.cat([har_spec, har_phase], dim=1)
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        rec.put(f"gen_stage{i}_leaky", x)
        x_source = gen.noise_convs[i](har)
        rec.put(f"gen_stage{i}_noise_conv", x_source)
        x_source = gen.noise_res[i](x_source, s)
        rec.put(f"gen_stage{i}_noise_res", x_source)
        x = gen.ups[i](x)
        rec.put(f"gen_stage{i}_ups", x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
            rec.put(f"gen_stage{i}_pad", x)
        x = x + x_source
        rec.put(f"gen_stage{i}_sum", x)
        xs = None
        for j in range(gen.num_kernels):
            block_out = gen.resblocks[i * gen.num_kernels + j](x, s)
            rec.put(f"gen_stage{i}_resblock{j}", block_out)
            if xs is None:
                xs = block_out
            else:
                xs += block_out
        x = xs / gen.num_kernels
        rec.put(f"gen_stage{i}_mrf", x)
    x = F.leaky_relu(x)
    rec.put("gen_post_leaky", x)
    x = gen.conv_post(x)
    rec.put("gen_conv_post", x)
    spec = torch.exp(x[:, :gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, gen.post_n_fft // 2 + 1:, :])
    rec.put("gen_spec", spec)
    rec.put("gen_phase", phase)
    return gen.stft.inverse(spec, phase)


def phoneme_ids(vocab, phonemes):
    """KModel.forward's mapping: drop unknown symbols, wrap in 0 (pad) ids."""
    ids = [vocab[p] for p in phonemes if vocab.get(p) is not None]
    return [0, *ids, 0]


def write_reference(rec, stem, meta, skipped=()):
    """Writes <stem>.safetensors (float32 stages minus `skipped`) and <stem>.json (metadata)."""
    stages = collections.OrderedDict((k, v) for k, v in rec.stages.items() if k not in skipped)
    meta = dict(meta)
    meta["stages"] = [{"name": k, "shape": list(v.shape)} for k, v in stages.items()]
    meta["skipped_stages"] = list(skipped)
    save_file(dict(stages), stem + ".safetensors", metadata={"format": "pt"})
    with open(stem + ".json", "w", encoding="utf-8") as out:
        json.dump(meta, out, indent=1, ensure_ascii=False)
        out.write("\n")


def write_wav(path, audio):
    """24 kHz mono int16 PCM."""
    pcm = (np.clip(audio.numpy(), -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def run_real(args):
    from kokoro import KModel, KPipeline
    snapshot = args.snapshot
    config_path = os.path.join(snapshot, "config.json")
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path,
                   model=os.path.join(snapshot, "kokoro-v1_0.pth")).eval()
    with open(config_path, encoding="utf-8") as src:
        config = json.load(src)
    g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
    results = list(g2p(REAL_TEXT))
    assert len(results) == 1, "sentence split into several chunks"
    phonemes = results[0].phonemes
    ids = phoneme_ids(model.vocab, phonemes)
    assert len(ids) == len(phonemes) + 2, "a phoneme is missing from the vocab"
    pack = torch.load(os.path.join(snapshot, "voices", VOICE + ".pt"), map_location="cpu", weights_only=True)
    ref_s = pack[len(phonemes) - 1]
    input_ids = torch.LongTensor([ids])
    parts = {n: getattr(model, n) for n in PART_NAMES}
    with torch.no_grad():
        rec = StageRecorder()
        audio, pred_dur = staged_forward(parts, input_ids, ref_s, config["style_dim"], rec)
        audio_again, _ = staged_forward(parts, input_ids, ref_s, config["style_dim"], StageRecorder())
        package_audio, package_dur = model.forward_with_tokens(input_ids, ref_s)
    assert torch.equal(audio, audio_again), "two zero-noise runs differ"
    assert torch.equal(pred_dur, package_dur)
    package_diff = (audio - package_audio.squeeze()).abs().max().item()
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "kokoro82m_ref")
    write_reference(rec, stem, {
        "text": REAL_TEXT, "phonemes": phonemes, "input_ids": ids,
        "voice": VOICE, "voice_row": len(phonemes) - 1,
        "pred_dur": pred_dur.tolist(), "sample_rate": SAMPLE_RATE,
        "noise": "zero", "speed": 1.0, "style_dim": config["style_dim"],
        "max_abs_diff_vs_KModel_forward_with_tokens": package_diff,
        "kokoro_version": kokoro.__version__, "torch_version": torch.__version__,
        "snapshot": snapshot})
    write_wav(stem + ".wav", audio)
    print(f"phonemes {phonemes!r}\ninput_ids {ids}\npred_dur {pred_dur.tolist()}")
    print(f"audio samples {audio.numel()}, |staged - KModel| max {package_diff:g}, determinism OK")
    print(f"wrote {stem}.safetensors/.json/.wav")


# Tiny model: real geometry (n_fft 20, hop 5, ups k20 s10 / k12 s6, resblock
# k3/7/11 d1/3/5), tiny widths. max_dur 3 keeps the audio short; seed 4 with
# these phonemes gives mixed durations [1,1,2,2,2,2] and 60% voiced samples.
TINY_SEED = 4
TINY_PHONEMES = "hˈO."
TINY_VOICE_ROWS = 32
# Derivable from kept stages (F0_pred; gen_spec/gen_phase), left out to keep the fixture small.
TINY_SKIPPED_STAGES = ("gen_f0_upsampled", "gen_uv", "gen_conv_post")
TINY_SIZES = {
    "hidden_dim": 8, "style_dim": 8, "max_dur": 3, "n_layer": 3,
    "embedding_size": 8, "bert_hidden": 16, "bert_heads": 2, "bert_ffn": 32,
    "bert_layers": 3, "bert_positions": 32,
    "upsample_initial_channel": 8, "decoder_width": 8, "asr_res_width": 4,
}


def tiny_config(real_config):
    """Real config.json schema with tiny sizes; plbert gains embedding_size."""
    z = TINY_SIZES
    config = json.loads(json.dumps(real_config))
    config["hidden_dim"] = z["hidden_dim"]
    config["style_dim"] = z["style_dim"]
    config["max_dur"] = z["max_dur"]
    config["n_layer"] = z["n_layer"]
    config["max_conv_dim"] = z["hidden_dim"]
    config["istftnet"]["upsample_initial_channel"] = z["upsample_initial_channel"]
    config["plbert"] = {
        "hidden_size": z["bert_hidden"], "num_attention_heads": z["bert_heads"],
        "intermediate_size": z["bert_ffn"], "max_position_embeddings": z["bert_positions"],
        "num_hidden_layers": z["bert_layers"], "dropout": 0.1,
        "embedding_size": z["embedding_size"]}
    return config


def build_tiny_parts(config):
    """Same constructor calls as KModel.__init__, with TinyDecoder for Decoder."""
    bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
    return {
        "bert": bert,
        "bert_encoder": nn.Linear(bert.config.hidden_size, config["hidden_dim"]),
        "predictor": ProsodyPredictor(style_dim=config["style_dim"], d_hid=config["hidden_dim"],
                                      nlayers=config["n_layer"], max_dur=config["max_dur"],
                                      dropout=config["dropout"]),
        "text_encoder": TextEncoder(channels=config["hidden_dim"],
                                    kernel_size=config["text_encoder_kernel_size"],
                                    depth=config["n_layer"], n_symbols=config["n_token"]),
        "decoder": TinyDecoder(dim_in=config["hidden_dim"], style_dim=config["style_dim"],
                               dim_out=config["n_mels"], decoder_width=TINY_SIZES["decoder_width"],
                               asr_res_width=TINY_SIZES["asr_res_width"], **config["istftnet"]),
    }


def instance_norm_keys(part):
    """AdaIN1d InstanceNorm affine keys: built by the package, absent from the real file."""
    return {f"{name}.{leaf}" for name, module in part.named_modules()
            if isinstance(module, nn.InstanceNorm1d) for leaf in ("weight", "bias")}


def randomize_tiny(parts, generator):
    """O(1)-scale random weights so no stage is near-identity; F0_proj puts F0
    across the 10 Hz voiced threshold so uv has both values."""
    def normal(shape, std, mean=0.0):
        return torch.randn(shape, generator=generator) * std + mean

    for part in parts.values():
        skipped = instance_norm_keys(part)
        for name, param in part.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if name in skipped:
                continue
            if leaf == "weight_g" or ".alpha" in name:
                param.data = torch.rand(param.shape, generator=generator) + 0.5
            elif leaf == "gamma" or (leaf == "weight" and param.dim() == 1):
                param.data = normal(param.shape, 0.1, 1.0)
            elif param.dim() == 1:
                param.data = normal(param.shape, 0.1)
            elif "embeddings" in name or name.startswith("embedding"):
                param.data = normal(param.shape, 1.0)
            else:
                param.data = normal(param.shape, 1.0 / math.sqrt(param[0].numel()))
    f0_proj = parts["predictor"].F0_proj
    f0_proj.weight.data = normal(f0_proj.weight.shape, 80.0 / math.sqrt(f0_proj.weight[0].numel()))
    f0_proj.bias.data = torch.tensor([30.0])


def checkpoint_state_dicts(parts, real_keys):
    """Per-part `module.`-prefixed OrderedDicts with DataParallel-style _metadata,
    like kokoro-v1_0.pth; the key sets must equal the real file's."""
    result = {}
    for part_name in PART_NAMES:
        part = parts[part_name]
        skipped = instance_norm_keys(part)
        state = part.state_dict()
        prefixed = collections.OrderedDict(
            ("module." + k, v.detach().clone().contiguous()) for k, v in state.items() if k not in skipped)
        assert set(prefixed) == real_keys[part_name], (
            part_name, sorted(set(prefixed) ^ real_keys[part_name])[:6])
        metadata = collections.OrderedDict([("", {"version": 1}), ("module", {"version": 1})])
        for key, value in state._metadata.items():
            if key:
                metadata["module." + key] = value
        prefixed._metadata = metadata
        result[part_name] = prefixed
    return result


def run_tiny(args):
    with open(os.path.join(args.snapshot, "config.json"), encoding="utf-8") as src:
        real_config = json.load(src)
    real_state = torch.load(os.path.join(args.snapshot, "kokoro-v1_0.pth"), map_location="cpu",
                            weights_only=True, mmap=True)
    real_keys = {n: set(real_state[n]) for n in PART_NAMES}
    del real_state
    config = tiny_config(real_config)
    torch.manual_seed(TINY_SEED)
    generator = torch.Generator().manual_seed(TINY_SEED)
    parts = build_tiny_parts(config)
    for part in parts.values():
        part.eval()
    randomize_tiny(parts, generator)
    style_dim = config["style_dim"]
    voice_pack = torch.randn((TINY_VOICE_ROWS, 1, 2 * style_dim), generator=generator)
    ids = phoneme_ids(config["vocab"], TINY_PHONEMES)
    assert len(ids) == len(TINY_PHONEMES) + 2, "a tiny phoneme is missing from the vocab"
    ref_s = voice_pack[len(TINY_PHONEMES) - 1]
    input_ids = torch.LongTensor([ids])
    rec = StageRecorder()
    audio, pred_dur = staged_forward(parts, input_ids, ref_s, style_dim, rec)
    audio_again, _ = staged_forward(parts, input_ids, ref_s, style_dim, StageRecorder())
    assert torch.equal(audio, audio_again), "two zero-noise runs differ"
    durations = pred_dur.tolist()
    voiced_fraction = rec.stages["gen_uv"].mean().item()
    print(f"input_ids {ids}\npred_dur {durations}\naudio samples {audio.numel()}, voiced {voiced_fraction:.2f}")
    assert len(set(durations)) > 1 and max(durations) > 1, durations
    assert 0.1 < voiced_fraction < 0.9, voiced_fraction

    folder = os.path.join(FIX, "tiny_kokoro82m")
    os.makedirs(os.path.join(folder, "voices"), exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as out:
        json.dump(config, out, indent=2, ensure_ascii=False)
        out.write("\n")
    # torch.save names the zip's top folder after the file stem; the real file's is "model".
    model_path = os.path.join(folder, "model.pth")
    torch.save(checkpoint_state_dicts(parts, real_keys), model_path)
    os.replace(model_path, os.path.join(folder, "kokoro-v1_0.pth"))
    torch.save(voice_pack.clone(), os.path.join(folder, "voices", VOICE + ".pt"))
    reloaded = build_tiny_parts(config)
    for name, state in torch.load(os.path.join(folder, "kokoro-v1_0.pth"), weights_only=True).items():
        reloaded[name].load_state_dict({k[7:]: v for k, v in state.items()}, strict=False)
        reloaded[name].eval()
    audio_reloaded, _ = staged_forward(reloaded, input_ids, ref_s, style_dim, StageRecorder())
    assert torch.equal(audio, audio_reloaded), "reloaded tiny checkpoint differs"
    write_reference(rec, os.path.join(FIX, "tiny_kokoro82m_ref"), {
        "phonemes": TINY_PHONEMES, "input_ids": ids, "voice": VOICE,
        "voice_row": len(TINY_PHONEMES) - 1, "pred_dur": durations,
        "sample_rate": SAMPLE_RATE, "noise": "zero", "speed": 1.0, "style_dim": style_dim,
        "seed": TINY_SEED, "voiced_fraction": voiced_fraction,
        "decoder_width": TINY_SIZES["decoder_width"], "asr_res_width": TINY_SIZES["asr_res_width"],
        "kokoro_version": kokoro.__version__, "torch_version": torch.__version__},
        TINY_SKIPPED_STAGES)
    print(f"wrote {folder}/ and {FIX}/tiny_kokoro82m_ref.safetensors/.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("real", "tiny"))
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="HF snapshot folder (config.json, kokoro-v1_0.pth, voices/)")
    parser.add_argument("--out", default=DEFAULT_REAL_OUT, help="real mode output folder")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.mode == "real":
        run_real(args)
    else:
        run_tiny(args)


if __name__ == "__main__":
    sys.exit(main())
