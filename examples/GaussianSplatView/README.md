# GaussianSplatView: Orbit Renders of a 3D Gaussian Splatting Scene

Renders a trained 3D Gaussian Splatting scene (Kerbl et al., SIGGRAPH 2023,
[*3D Gaussian Splatting for Real-Time Radiance Field
Rendering*](https://arxiv.org/abs/2308.04079)) from a camera that orbits the
scene centre, and writes the views as PNG frames and one animated GIF.

**Status: coded, not yet run on a real INRIA scene.** The rasterizer is tested
on hand-derived cases (`TestNeuralGaussianSplat`), the OpenCL rasterizer
against the CPU one (`TestOpenCLMatchesCPU`, max |difference| below 1e-4), and
the example runs end to end with `--demo` on the CPU and on PoCL. No trained
`point_cloud.ply` has been rendered through it yet.

## How it works

```
point_cloud.ply (INRIA layout)
  -> TGaussianSplatScene.LoadFromPLY     sigmoid opacity, exp scale, unit quaternion, SH 0-3
  -> per orbit frame: SetGaussianSplatCameraLookAt(eye on the orbit, orbit target)
  -> TGaussianSplatRasterizer.Render
       Preprocess   per Gaussian: 2D conic, 3-sigma radius, SH colour, depth, tile rect
       BinTiles     one depth sort, then per 16x16 tile lists, nearest first
       Blend        front-to-back alpha blending per pixel
  -> frame_NNN.png + orbit.gif (TNNetAnimatedGIFWriter)
```

With `--gpu`, `TGaussianSplatRasterizerCL` runs `Preprocess` in the
`cai_gaussian_splat_preprocess` kernel (one work-item per Gaussian) and
`Blend` in `cai_gaussian_splat_blend` (one 16x16 work-group per tile; the
tile's Gaussians go through local memory 256 at a time, as in the reference
CUDA forward). `BinTiles` stays on the host: `TGaussianSplatRasterizerCL.Preprocess`
downloads the radii, depths and tile rects, `BinTiles` builds the per-tile
lists, and `TGaussianSplatRasterizerCL.Blend` uploads them. The means, conics
and colours stay in OpenCL memory, and the scene is uploaded once.

The orbit turns about the up axis through the orbit target. Real scenes have
far background Gaussians that would pull a plain centroid off the object, so
the target is robust to them:

- Without `--cameras`, `TGaussianSplatScene.ComputeRobustCentre` takes the
  per-axis median of the Gaussian means, each weighted by its opacity;
  Gaussians with opacity below 0.1 do not vote. The up axis is world -y
  (COLMAP axes: image +y is down), and the orbit starts on the -z side of the
  target at 3 times the opacity-weighted median distance of the means from
  the target.
- With `--cameras`, `GaussianSplatNearestPointToRays` takes the least-squares
  point nearest to the view axes of all cameras in `cameras.json`: the point
  the capture cameras look at. When the axes are near-parallel or the point
  lies behind at least half of the cameras (a forward-facing capture), the
  weighted median above stays the target. The orbit starts at camera 0, so
  the radius is its distance from the target.

`--radius` overrides the radius in both cases.

## Run

```
../../bin/x86_64-linux/bin/GaussianSplatView --demo --orbit 12 --size 160x120
../../bin/x86_64-linux/bin/GaussianSplatView --ply output/garden/point_cloud/iteration_30000/point_cloud.ply \
  --cameras output/garden/cameras.json --orbit 60 --size 640x480 --gpu --out garden_orbit
```

| option | meaning | default |
| --- | --- | --- |
| `--ply FILE` | INRIA `point_cloud.ply` (binary or ascii; SH degree 0-3) | required unless `--demo` |
| `--demo` | render a random scene of 600 Gaussians instead | off |
| `--orbit N` | frames in one full orbit | 36 |
| `--size WxH` | image size in pixels | 320x240 |
| `--fov DEG` | horizontal field of view, square pixels | 60 |
| `--cameras FILE` | INRIA `cameras.json`: camera 0 gives `fx`/`fy` (scaled to `--size`), the up axis (minus its y axis) and the orbit start (its position); all cameras give the orbit target (see above) | off |
| `--radius R` | orbit radius in scene units | see above |
| `--gpu` | preprocess and blend in OpenCL (first platform, first device) | off |
| `--out DIR` | output folder | `gaussian_splat_frames` |

## Memory

A scene keeps 3 + 3 + 4 + 1 + 3 x (SH coefficients) floats per Gaussian:
62 floats at SH degree 3, so about 1.5 GB for 6 million Gaussians, plus the
per-frame arrays of the rasterizer. Large scenes need a machine with RAM (and,
with `--gpu`, OpenCL memory) to match.

## Build

```
lazbuild -B GaussianSplatView.lpi
```

The binary goes to `bin/x86_64-linux/bin/`; `--gpu` finds `neural/neural.cl`
relative to the binary or to this folder.
