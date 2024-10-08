# Training a Simple Neural Network Model for Text Generation

## Creating a Vocabulary and Tokenizing a Dataset
You can create your own vocabulary with CAI’s tokenizer from the unit [neuraltokenizer.pas](https://github.com/joaopauloschuler/neural-api/blob/master/neural/neuraltokenizer.pas). A vocabulary with 3000 tokens can be created with:
```
var
  X: TNeuralTokenizer;
begin
  X := TNeuralTokenizer.Create;
  X.FitOnFile('datasets/tinystories-10k.txt', 3000);
  X.SaveToFile('datasets/tinystories-vocab-3k-cai.txt');
  X.Free;
end;
```
The dataset can then be tokenized with:
```
var
  X: TNeuralTokenizer;
begin
  X := TNeuralTokenizer.Create;
  X.LoadVocabularyFromFile('datasets/tinystories-vocab-3k-cai.txt');
  X.TokenizeFileToCsv('datasets/tinystories-100k.txt','datasets/tinystories-100k-tokenized3k.csv');
  X.Free;
end;
```
Above files can be found at the [TinyStories4Pascal-Tokenized-v2](https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2) repository.

## Samplers
Samplers are used to probabilistically select the next token (character) from the probabilities guessed by the neural network. The Greedy, Top-K, and Top-P samplers provide different ways to predict the next character in a sequence.

Greedy Sampling:
* Always selects the token with the highest probability at each step.
* Tends to produce repetitive and deterministic output.

Top-K Sampling:
* Samples from the K most likely next tokens at each step.
* K is a parameter that controls diversity - a bigger K leads to more diverse results.

Top-P Sampling:
* Samples from the smallest possible set of tokens whose cumulative probability exceeds P at each step.
* P is a parameter between 0 and 1 controlling diversity - lower P produces less diversity.

In summary:
Greedy sampling takes the most likely token, leading to less diversity. Top-K and Top-P allow controlling diversity by adjusting their parameters.

These samplers are available in plain pascal code:

```
  { TNNetSamplerGreedy }
  TNNetSamplerGreedy = class (TNNetSamplerBase)
    public
      function GetToken(Origin: TNNetVolume): integer; override;
  end;

  { TNNetSamplerTopK }
  TNNetSamplerTopK = class (TNNetSamplerBase)
    protected
      FTopK: integer;
    public
      constructor Create(TopK: integer);
      function GetToken(Origin: TNNetVolume): integer; override;
  end;

  { TNNetSamplerTopP }
  TNNetSamplerTopP = class (TNNetSamplerBase)
    protected
      FTopP: TNeuralFloat;
    public
      constructor Create(TopP: TNeuralFloat);
      function GetToken(Origin: TNNetVolume): integer; override;
  end;
``` 

In this source code example, the sampler is created with  `FSampler := TNNetSamplerTopP.Create(0.4);`

Then, you can just call the following to see the magic:

```
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'one day', FSampler),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'once upon', FSampler),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'billy', FSampler),'.');
```

The loading and saving of neural networks (NN) can be done with:
```
   NN := TNNet.Create;
   NN.LoadFromFile('MyTrainedNeuralNetwork.nn');
   NN.SaveToFile('MyTrainedNeuralNetwork.nn');
```

A small chat bot can be coded with:

```
procedure TestFromFile;
var
  S: string;
  oSampler: TNNetSamplerBase;
  NN: TNNet;
begin
  oSampler := TNNetSamplerTopP.Create(0.6);
  NN := TNNet.Create();
  WriteLn('Loading neural network.');
  NN.LoadFromFile(csAutosavedFileName);
  NN.DebugStructure();
  WriteLn();
  WriteLn('Write something and I will reply.');
  repeat
    Write('User: ');
    ReadLn(S);
    WriteLn('Neural network: ',GenerateStringFromCasualNN(NN, FDictionary, LowerCase(S), oSampler),'.');
  until S = 'exit';
  NN.Free;
  oSampler.Free;
end;
```

## Example with Transformer Decoder, Vocabulary and Multiple Outputs
The currently leading NLP neural network models use tokenized datasets and vocabulary. In this example, the [tokenized dataset](https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2) and its vocabulary can be downloaded with:
```
git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2
unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip
unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip
```

Plenty of models are constructed via a stack of transformer decoder modules. This stack can be created with:
```
for I := 1 to 2 do FNN.AddTransformerBlockCAI(8, 2048, true, false, false);
```

Finally, an output layer with one output per input in the context can be added with:
```
    FNN.AddLayer([
      TNNetPointwiseConvLinear.Create(csModelVocabSize),
      TNNetPointwiseSoftMax.Create(1)
    ]);
```
The full source code can be found at [NLP with Vocabulary](https://colab.research.google.com/github/joaopauloschuler/neural-api/blob/master/examples/SimpleNLP/TrasformerDecodersWithTokenizer.ipynb).

After training, this model produces sentences such as:
* one day, a little girl named lucy went to the park with her mom. lucy saw a big tree with a hole in it. she wanted to climb the tree and see what was on the other .
* once upon a time, there was a little girl named lily. she loved to play with her toys and her favorite toy was a teddy bear. one day, lily's mom asked her to help with.
* billy was a little boy who loved to play outside. one day, he was playing in the park when he saw a big puddle in the grass. he ran over to it and started to .

An advanced source code example can be found [here](https://github.com/joaopauloschuler/gpt-3-for-pascal) .
