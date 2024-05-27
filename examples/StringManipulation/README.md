# Predicting the next character in a string

In this source code example, to make things very simple, the dataset has 3 strings:
```
    FDataset[0] := 'happy good morning.'+chr(1);
    FDataset[1] := 'fantastic good evening.'+chr(1);
    FDataset[2] := 'superb good night.'+chr(1);
```

The neural network is built with:

```
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetPointwiseConvReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectLinear.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);
```

The constants above are defined with:
```
const
  csContextLen = 64;  // The input string can have up to 64 characters.
  csVocabSize  = 128; // Character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.
```

After training the NN, we test with:

```
    WriteLn('Testing.');
    WriteLn(GenerateStringFromChars(FNN, 'happy'));
    WriteLn(GenerateStringFromChars(FNN, 'fantastic'));
    WriteLn(GenerateStringFromChars(FNN, 'superb'));
```

Then NN gets it 100% right with the following output:
```
happy good morning.
fantastic good evening.
superb good night.
```

## How does it work?
Each character is encoded into a number from 0 to 127. This number from 0 to 127 is then transformed into a vector with 128 elements where only one element values 1. This is called “one-hot encoding” (https://en.wikipedia.org/wiki/One-hot). 
This is why the input is defined with (csContextLen, 1, csVocabSize). Then, to decrease the dimensionality from 128 dimensions in the vector, a pointwise convolution is used (TNNetPointwiseConvReLU). 
Each input character is converted into an 128 elements vector in reverse order so “good” will become “doog”. The last layer of the NN has 128 elements so the element with highest value (or probability) is the next predicted character.

The one-hot encoding is done with one API call:
```
procedure TVolume.OneHotEncodingReversed(aTokens: string); overload;
```

In the case that your input is not a string, you could call the following instead:

```
procedure TVolume.OneHotEncoding(aTokens: array of integer); overload;
```

So, for each character predicted, this character is added to the input string and the process is repeated until a termination token is found (chr(0) or chr(1)). This is how calling GenerateStringFromChars(FNN, 'superb') outputs ‘superb good night’.
