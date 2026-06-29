# CLIPSegPrompt: text-prompted zero-shot segmentation

The demo for `BuildCLIPSegFromSafeTensors` (`neural/neuralpretrained.pas`), a
**free-text prompt → dense single-channel mask** importer. CLIPSeg (Lüddecke &
Ecker 2022, *Image Segmentation Using Text and Image Prompts*,
[arXiv:2112.10003](https://arxiv.org/abs/2112.10003); `CIDAS/clipseg-rd64-refined`)
runs an image through a **frozen CLIP ViT**, taps a few intermediate encoder
layers (`config.extract_layers`), **FiLM-modulates** them with the CLIP **text**
embedding of an arbitrary prompt, refines through a small post-norm transformer
decoder and upsamples (a single ConvTranspose2d) to an `H×W` logit map: the mask
for "whatever the prompt names", with **no fixed label set**.

Heavy reuse: the frozen CLIP vision and text towers are the same pre-LN CLIP
encoder blocks as `BuildClipFromSafeTensors`; the new code is only the
FiLM-conditioned decoder (`TNNetFiLM` + a post-norm CLIP-style block + a
DepthToSpace transposed-conv upsample).

## Build / run

```
cd examples/CLIPSegPrompt
lazbuild CLIPSegPrompt.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/CLIPSegPrompt
../../bin/x86_64-linux/bin/CLIPSegPrompt model.safetensors [config.json]
```

## Input

The real `CIDAS/clipseg-rd64-refined` checkpoint is large and not obtainable
offline, so — exactly like the repo's NAFNet/SwinIR pico fixtures — this falls
back to the committed config-faithful **random** pico CLIPSeg
(`tests/fixtures/tiny_clipseg.safetensors` + `..._config.json`,
parity-checked < 1e-4 in `TestCLIPSegParity`). The pico net is random (not
trained), so this is a **wiring/throughput smoke**: a deterministic synthetic
image is generated in-code and the prompt is supplied as hand-typed **token ids**
(the pico fixture has no tokenizer). Pass a real `.safetensors` (+ `config.json`
sibling) and your CLIP tokenizer's ids to segment with a trained checkpoint.

## Output

Writes `clipseg_image.ppm` (the synthetic input) and `clipseg_mask.ppm` (the
logit map thresholded at 0 → binary mask), and prints the config plus an ASCII
preview of the image and the mask. Pure CPU, well under a second on the fixture.
