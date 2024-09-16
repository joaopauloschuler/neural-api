# Training a Simple Neural Network Model for Text Generation
## The Simple Example
This source code example shows a (hello world) small neural network trained on the [Tiny Stories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). This code

```
    WriteLn(GenerateStringFromChars(NFit.NN, 'once', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'one ', FSampler),'.');
```

produces this output:
```
once upon a time, there was a little girl named lily. she loved to play outside i.
one day, a little girl named lily was playing in her garden. she saw a big car wi.
```

You can find the raw training file and run by yourself at:
https://colab.research.google.com/github/joaopauloschuler/neural-api/blob/master/examples/SimpleNLP/NLP_CAI_TinyStories_Simple_Example.ipynb

### Details
This source code above uses a neural network to guesses the next character in a string.
It downloads the [Tiny Stories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) and trains a small Pascal written neural network model. The neural network model is built with:

```
const
  csContextLen = 81;
  csTrainingFileName = 'tinystories.txt';
  csVocabSize  = 128; // Character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.
...
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetPointwiseConv.Create(32,1),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(64,3,0,1,1),
      TNNetMaxPool.Create(3),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(128*3,3,0,1,1),
      TNNetPointwiseConvReLU.Create(1024,0),
      TNNetMaxPoolWithPosition.Create(27,27,0,1,0),
      TNNetPointwiseConvReLU.Create(1024),
      TNNetPointwiseConvReLU.Create(128),
      TNNetFullConnectLinear.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);
```

This neural network has some characteristics:
* It’s character based. Therefore, there is no dictionary. The convolutional layers are responsible for learning the words. In the first epochs of the training, we can see that the neural network is learning the words. This architecture benefits from the small vocabulary found in the “Tiny Stories” dataset.
* It predicts the next character in an input sequence (or context). In this example, the context is 81 characters.
* There is no recursive computation. It’s a convolutional model. Therefore, it’s memory efficient and can be computed in a highly parallel environment.
* One of the max pooling layers inserts the positional information of the max values.
* In this particular example, it learns very well the [Tiny Stories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). The very same model was used to train with wikipedia but wikipedia vocabulary and sentence structures are too complex for this small 2.8 million parameters model. You can just replace tinystories.txt and train it on your own text file (dataset). This source code is the “hello world” of the NLP. Don’t expect too much from it.

In the case that you are curious, there are plenty of scientific studies supporting NLP with CNNs:
* https://aclanthology.org/W18-6127/ - Convolutions Are All You Need (For Classifying Character Sequences)
* https://arxiv.org/abs/1712.09662 - CNN Is All You Need
* https://arxiv.org/abs/1804.09541 - QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension
* https://aclanthology.org/N19-1407.pdf - Convolutional Self-Attention Networks
* https://arxiv.org/pdf/1805.08318.pdf - Self-Attention Generative Adversarial Networks

### API Methods
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
    WriteLn(GenerateStringFromChars(NFit.NN, 'once', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'one ', FSampler),'.');
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
    WriteLn('Neural network: ',GenerateStringFromChars(NN, LowerCase(S), oSampler),'.');
  until S = 'exit';
  NN.Free;
  oSampler.Free;
end;
```
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

## Example with Transformer Decoder, Vocabulary and Multiple Outputs
The currently leading NLP neural network models use tokenized datasets and vocabulary. The tokenized dataset and the vocabulary can be downloaded with:
```
git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal
unzip TinyStories4Pascal/tinystories_tokenized_81.csv.zip
unzip TinyStories4Pascal/tinystories_vocab.csv.zip
```

Plenty of models are constructed via a stack of transformer decoder modules. This stack can be created with:
```
for I := 1 to 2 do FNN.AddTransformerBlockCAI(8, 2048, true, false, false);
```

Finally, an output layer with one output per input in the context can be added with:
```
    FNN.AddLayer([
      TNNetPointwiseConvLinear.Create(csEmbedDim),
      TNNetPointwiseConvLinear.Create(csModelVocabSize),
      TNNetPointwiseSoftMax.Create(1)
    ]);
```
The full source code can be found at [NLP with Vocabulary](https://colab.research.google.com/github/joaopauloschuler/neural-api/blob/master/examples/SimpleNLP/transformer_decoder_tiny_stories_dataset_3k_vocab.ipynb).

After training, this model produces sentences such as:
* one day , a little boy named tim went to the park with his mom . they saw a big tree with many leaves . tim wanted to climb the tree , but his mom said , " no , tim , you can ' t climb the tree . it ' s not safe . " tim was sad and said , " but i want to climb the tree . " his mom said , " don '.
* once upon a time , there was a little girl named lily . she loved to play outside in the sunshine . one day , she saw a big , scary dog . the dog looked scary and lily was scared . she ran to her mommy and said , " mommy , there ' s a big dog outside ! " her mommy said , " don ' t worry , lily . the dog is friendly and friendly.
* billy was a little boy who loved to play . he had a lot of toys , but his favorite was a big red ball . he would throw it up and catch it , and he would throw it . one day , billy was playing with his ball in the park . he saw a big , shiny red ball and he wanted to play with it . he ran to the ball and tried to grab it.


