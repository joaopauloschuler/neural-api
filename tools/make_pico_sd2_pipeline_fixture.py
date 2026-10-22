#!/usr/bin/env python3
"""Generate the pico Stable Diffusion 2.x pipeline fixture for
TStableDiffusionPipeline: a tiny RANDOM-weight folder shaped like
stabilityai/stable-diffusion-2-1 (nothing downloaded):

  tests/fixtures/tiny_sd2/
    model_index.json                     StableDiffusionPipeline
    scheduler/scheduler_config.json      DDIMScheduler, prediction_type
                                         v_prediction, skip_prk_steps (as 2.1)
    text_encoder/{config.json,model.safetensors}   CLIPTextModel, hidden_act
                                         "gelu" (OpenCLIP), keys under text_model.
    tokenizer/{vocab.json,merges.txt,tokenizer_config.json,
               special_tokens_map.json}  tiny_sd15's BPE; tokenizer_config says
                                         pad "<|endoftext|>", special_tokens_map
                                         says "!" and wins (as in 2.1)
    unet/{config.json,diffusion_pytorch_model.safetensors}  use_linear_projection,
                                         a per-block head list, upcast_attention,
                                         sample_size SD2_SAMPLE_SIZE
    vae/{config.json,diffusion_pytorch_model.safetensors}   2 blocks (factor 2)

The oracle files come from the tiny_sd15 generators run on this folder (module
globals repointed), diffusers float64, latents passed explicitly:

  tests/fixtures/tiny_sd2_pipeline_io.json   tokens, embeds, DDIM run, image
  tests/fixtures/tiny_sd2_samplers_io.json   PNDM, Euler, DPM++ 2M
  tests/fixtures/tiny_sd2_lcm_io.json        LCMScheduler text2img + img2img
  tests/fixtures/tiny_sd2_img2img_io.json    img2img: DDIM, PNDM, DPM++ 2M,
                                             Euler, Euler a
(layouts as their tiny_sd15 counterparts; see the two tiny_sd15 generators).

Coded by Claude (AI).

Usage (from the repo root, inside the `x` venv):
  python tools/make_pico_sd2_pipeline_fixture.py
"""
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from diffusers import (AutoencoderKL, DDIMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import make_pico_sd_pipeline_fixture as pipeline_fixture
import make_pico_sd_conditioning_fixtures as conditioning_fixture

FIXTURES = pipeline_fixture.FIXTURES
SD2_DIR = os.path.join(FIXTURES, "tiny_sd2")
SD2_SEED = 20261022
SD2_SAMPLE_SIZE = 12     # latent grid; the default image is 12 * 2 = 24
TEXT_HIDDEN = 32


def write_sd2_tokenizer_config(folder):
    """stabilityai/stable-diffusion-2-1 tokenizer/: the two files disagree on
    the pad token; transformers takes special_tokens_map.json's "!"."""
    def added(content):
        return {"__type": "AddedToken", "content": content, "lstrip": False,
                "normalized": True, "rstrip": False, "single_word": False}
    config = {"add_prefix_space": False, "bos_token": added("<|startoftext|>"),
              "do_lower_case": True, "eos_token": added("<|endoftext|>"),
              "errors": "replace", "model_max_length": 77,
              "pad_token": "<|endoftext|>", "tokenizer_class": "CLIPTokenizer",
              "unk_token": added("<|endoftext|>")}
    with open(os.path.join(folder, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    special = {"bos_token": added("<|startoftext|>"),
               "eos_token": added("<|endoftext|>"), "pad_token": "!",
               "unk_token": added("<|endoftext|>")}
    for value in special.values():
        if isinstance(value, dict):
            value.pop("__type")
    with open(os.path.join(folder, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)


def build_sd2_folder():
    torch.manual_seed(SD2_SEED)
    vocab, merges = pipeline_fixture.train_clip_bpe()
    text_config = CLIPTextConfig(
        vocab_size=len(vocab), hidden_size=TEXT_HIDDEN, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=77,
        hidden_act="gelu", layer_norm_eps=1e-5, projection_dim=TEXT_HIDDEN,
        bos_token_id=0, eos_token_id=2, pad_token_id=1)
    text_encoder = CLIPTextModel(text_config)
    pipeline_fixture.perturb(text_encoder)
    unet = UNet2DConditionModel(
        sample_size=SD2_SAMPLE_SIZE, in_channels=4, out_channels=4,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=1, cross_attention_dim=TEXT_HIDDEN,
        attention_head_dim=(2, 4), use_linear_projection=True,
        upcast_attention=True, norm_num_groups=4)
    pipeline_fixture.perturb(unet)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=SD2_SAMPLE_SIZE * 2, scaling_factor=0.18215)
    pipeline_fixture.perturb(vae)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
        steps_offset=1, prediction_type="v_prediction", timestep_spacing="leading")
    tokenizer_dir = os.path.join(SD2_DIR, "tokenizer")
    if os.path.isdir(SD2_DIR):
        shutil.rmtree(SD2_DIR)
    pipeline_fixture.write_tokenizer(tokenizer_dir, vocab, merges)
    write_sd2_tokenizer_config(tokenizer_dir)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    assert tokenizer.pad_token == "!" and tokenizer.pad_token_id == 0, tokenizer.pad_token
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    pipe.save_pretrained(SD2_DIR, safe_serialization=True)
    for name in os.listdir(tokenizer_dir):
        os.remove(os.path.join(tokenizer_dir, name))
    pipeline_fixture.write_tokenizer(tokenizer_dir, vocab, merges)
    write_sd2_tokenizer_config(tokenizer_dir)
    scheduler_file = os.path.join(SD2_DIR, "scheduler", "scheduler_config.json")
    with open(scheduler_file) as f:
        scheduler_config = json.load(f)
    scheduler_config["skip_prk_steps"] = True
    with open(scheduler_file, "w") as f:
        json.dump(scheduler_config, f, indent=2)
    te_file = os.path.join(SD2_DIR, "text_encoder", "model.safetensors")
    tensors = load_file(te_file)
    tensors = {(k if k.startswith("text_model.") else "text_model." + k): v
               for k, v in tensors.items()}
    save_file(tensors, te_file, metadata={"format": "pt"})


def point_generators_at_sd2():
    pipeline_fixture.PICO_DIR = SD2_DIR
    pipeline_fixture.IMAGE_SIZE = SD2_SAMPLE_SIZE * 2
    pipeline_fixture.IO_FILE = os.path.join(FIXTURES, "tiny_sd2_pipeline_io.json")
    pipeline_fixture.SAMPLERS_FILE = os.path.join(FIXTURES, "tiny_sd2_samplers_io.json")
    pipeline_fixture.LCM_FILE = os.path.join(FIXTURES, "tiny_sd2_lcm_io.json")
    conditioning_fixture.PICO_DIR = SD2_DIR
    conditioning_fixture.IMG2IMG_IO = os.path.join(FIXTURES, "tiny_sd2_img2img_io.json")


def check_folder():
    pipe = StableDiffusionPipeline.from_pretrained(SD2_DIR, torch_dtype=torch.float64,
                                                   safety_checker=None)
    assert pipe.tokenizer.pad_token == "!", pipe.tokenizer.pad_token
    assert pipe.scheduler.config.prediction_type == "v_prediction"
    assert pipe.text_encoder.config.hidden_act == "gelu"
    assert pipe.unet.config.use_linear_projection
    assert pipe.unet.config.sample_size * pipe.vae_scale_factor == SD2_SAMPLE_SIZE * 2
    ids = pipe.tokenizer("a photo", padding="max_length", max_length=77).input_ids
    assert ids[-1] == 0, ids


if __name__ == "__main__":
    build_sd2_folder()
    check_folder()
    point_generators_at_sd2()
    pipeline_fixture.make_oracle()
    pipeline_fixture.make_sampler_oracle()
    pipeline_fixture.make_lcm_oracle()
    conditioning_fixture.make_img2img()
