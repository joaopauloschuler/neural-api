# VideoPrediction — ConvLSTM next-frame prediction

A spatiotemporal (video) example: next-frame prediction in the Moving-MNIST style
(Srivastava et al. 2015) on a self-contained, download-free synthetic dataset.
The model watches the first N frames of a clip and must predict the (N+1)-th
frame. This is the showcase for **`TNNetConvLSTMCell`**, a convolutional LSTM
(Shi et al. 2015, *"Convolutional LSTM Network"*,
[arXiv:1506.04214](https://arxiv.org/abs/1506.04214)): the spatial, image-state
analogue of the dense recurrent cells (`TNNetMinLSTM` / `TNNetSLSTMCell` /
`TNNetMLSTMCell`). Instead of a vector hidden state it carries `(H,W,HiddenC)`
feature **maps** and replaces every gate matrix-multiply with a `K×K`
same-padding convolution over `[x_t ; h_{t-1}]` — exactly what lets the
recurrence track *where* the blob is and *where it is heading*.

## Data (synthetic moving blob, in-code)

Each clip is `N+1` frames of a `12×12` grayscale image. A small bright blob
starts at a random position with a random integer velocity (`vx,vy ∈ {-1,0,+1}`,
not both zero), moves one cell per frame, and reflects off the borders. Pixels are
in `[-1,1]` (background `-1`, blob centre `+1`) to match the model's Tanh output.
The first N frames are the input sequence; the final frame is the target. Frames
are packed along the X axis as `(N*GRID, GRID, 1)` — the layout
`TNNetConvLSTMCell` expects.

## Model

```
Input(N*GRID, GRID, 1, 1)               # N frames packed along X
  -> TNNetConvLSTMCell(N, HiddenC, 3)    # per-step hidden maps (N*GRID,GRID,HiddenC)
  -> TNNetCrop((N-1)*GRID, 0, GRID, GRID)# keep ONLY the last timestep's map
  -> TNNetConvolutionLinear(1, 3, 1, 1)  # 3x3 conv head -> 1 channel
  -> TNNetHyperbolicTangent              # predicted next frame in [-1,1]
```

Training is plain supervised regression: `Compute(inputFrames)` then
`Backpropagate(nextFrame)` (MSE on predicted vs true next frame).

## Build & run

```
lazbuild examples/VideoPrediction/VideoPrediction.lpi
./bin/x86_64-linux/bin/VideoPrediction          # smoke run (default; under 5 min on CPU)
./bin/x86_64-linux/bin/VideoPrediction --full   # more clips / epochs / hidden channels
```

## Output

- Held-out MSE/MAE before and after training (per epoch), plus a final
  improvement percentage vs the untrained baseline.
- An ASCII panel (`N input frames | PREDICTED | GROUND-TRUTH`) and
  `videoprediction_sample.ppm` (inputs gray | prediction green | truth red) for
  one held-out clip.
