# SelfTest — neural-api self-test harness

A small harness that runs the library's built-in test suites to sanity-check a
build of neural-api. It exercises three areas:

- `TestTNNetVolume` — the volumes (tensor) API.
- `TestKMeans` — K-means clustering.
- `TestConvolutionAPI` — the convolution API.

## Running

No arguments, no dataset, no download — entirely self-contained and CPU-based.

```
cd examples/SelfTest
# build with lazbuild SelfTest.lpi (or fpc), then:
./SelfTest
```

It prints `Testing Volumes API ...` and `Testing Convolutional API ...` around
the suite output, then `Press ENTER to quit.` and waits for input. The detailed
pass/fail output comes from the test functions themselves.

Coded by Joao Paulo Schwarz Schuler.
