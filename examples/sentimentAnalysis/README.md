# sentimentAnalysis — character-level CNN sentiment classifier (SST-2)

A nano **character-level convolutional** NLP classifier that learns binary
sentiment (negative / positive) on the [SST-2](https://huggingface.co/datasets/sst2)
movie-review dataset. No tokenizer or vocabulary file is needed — the input is a
raw ASCII character window, one-hot encoded.

## Data

Reads `sst2_local/sst2_train.txt`, one example per line in the format
`<label><space><text>` where `<label>` is `0` (negative) or `1` (positive). The
loader lowercases each line, splits off the leading class char, and keeps the
rest as the text. A validation split is carved out of the training set by
removing every 21st row (the commented-out path can instead load
`sst2_local/sst2_validation.txt`). Lines shorter than 3 characters are dropped.

Each training sample is the review text with ~10% of its characters randomly
removed (`RemoveRandomChars`) as augmentation, truncated to `csContextLen=81`
characters and one-hot encoded over a 128-symbol ASCII vocabulary via
`TNNetVolume.OneHotEncodingReversed`.

## The model

A 1-D character CNN over the `(81, 1, 128)` one-hot window:

```
Input(81, 1, 128)
 -> PointwiseConv(32, 1)
 -> Dropout(0.5)
 -> PadXY(1,0)
 -> ConvolutionReLU(64, 3)
 -> MaxPool(3)
 -> PadXY(1,0)
 -> ConvolutionReLU(128, 3)
 -> PointwiseConvReLU(1024, 1)
 -> MaxPoolWithPosition(27, 27)
 -> FullConnectLinear(2)
 -> SoftMax
```

The `TNNetMaxPoolWithPosition` layer pools the whole sequence while injecting
positional information, and the final `FullConnectLinear(2) -> SoftMax` produces
the two-class distribution.

## Training

Trained with `TNeuralDataLoadingFit` (`lr=0.001`, batch 32, 50 epochs,
`EnableClassComparison` + `EnableDefaultLoss`, `AvgWeightEpochCount=1`). After
every epoch the `OnAfterEpoch` callback runs a quick smoke test, classifying a
few fixed strings via `GetClassFromChars`.

## How to run

```
cd examples/sentimentAnalysis
fpc -O3 -Mobjfpc -Sh -Fu../../neural sentimentAnalysis.lpr
./sentimentAnalysis
```

(or open `sentimentAnalysis.lpi` in Lazarus). Make sure `sst2_local/sst2_train.txt`
is present first. Pure CPU; multi-threaded via `TNeuralDataLoadingFit`.

The trained network autosaves to `sentiment.nn`. An interactive REPL
(`TestFromFile`, currently commented out in the program body) can load that file
and classify free text typed by the user until `exit`.

## Expected output

The training log prints validation/test accuracy per epoch, and the
`OnAfterEpoch` smoke test prints lines such as:

```
"i hated this movie." is negative.
"horrible. i do not recommend." is negative.
```

confirming the classifier learns review polarity from raw characters.
