(*
neuraldatasets
Copyright (C) 2017 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuraldatasets;
{
  neuraldatasets Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/
}

interface

uses
  {$IFNDEF FPC}System.Classes, Windows, Vcl.Graphics,{$ENDIF}
  neuralvolume, neuralnetwork
  {$IFDEF FPC},
  FPimage, FPReadBMP, FPReadPCX, FPReadJPEG, FPReadPNG,
  FPReadGif, FPReadPNM, FPReadPSD, FPReadTGA, FPReadTiff,
  FPWriteBMP, FPWritePCX, FPWriteJPEG, FPWritePNG,
  FPWritePNM, FPWriteTGA, FPWriteTiff
  {$ENDIF}
  ;

{$include neuralnetwork.inc}

const
  csNeuralEncodingMethodInt = 0;
  csNeuralEncodingMethodOneHot = 1;
  csNeuralEncodingMethodGroupedOnHot = 2;
  csNeuralEncodingMethodIntChar = 3;

type
  TTinyImageChannel = packed array [0..31, 0..31] of byte;
  TTinyImageChannel1D = packed array [0..32 * 32 - 1] of byte;
  TMNistImage = packed array [0..27, 0..27] of byte;

  TTinyImage = packed record
    bLabel: byte;
    R, G, B: TTinyImageChannel;
  end;

  TCifar100Image = packed record
    bCoarseLabel: byte;
    bFineLabel: byte;
    R, G, B: TTinyImageChannel;
  end;

  TTInyImageFile = file of TTinyImage;
  TCifar100File = file of TCifar100Image;

  // Useful for gray scale image
  TTinySingleChannelImage = packed record
    bLabel: byte;
    Grey: TTinyImageChannel;
  end;

  TTinySingleChannelImage1D = packed record
    bLabel: byte;
    Grey: TTinyImageChannel1D;
  end;

  TTinySingleChannelImagePtr = ^TTinySingleChannelImage;
  TTinySingleChannelImage1DPtr = ^TTinySingleChannelImage1D;

const
  csTinyImageLabel: array[0..9] of string =
    (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
    );

  csMachineAnimalCifar10Pos: array[0..9] of integer =
  (
    0,
    1,
    8, // new bird position is 8. This position is now ship.
    9, // new cat position  is 9. This position is now truck.
    4,
    5,
    6,
    7,
    2, // new ship position is 2. This position is now bird.
    3  // new truck         is 3. This position is now cat.
  );

  csMachineAnimalCifar10Labels: array[0..9] of string =
  (
    'airplane',
    'automobile',
    'ship',  // used to be bird
    'truck', // used to be cat
    'deer',  // used to be deer
    'dog',   // used to be dog
    'frog',  // used to be frog
    'horse', // used to be horse
    'bird',  // used to be ship
    'cat'    // used to be truck
  );

type

  { TFileNameList }
  TFileNameList = class(TStringListInt)
    protected
      FClassCount: integer;
      {$IFDEF HASTHREADS}FCritSecLoad: TRTLCriticalSection;{$ENDIF}
    public
      constructor Create();
      destructor Destroy(); override;

      procedure GetImageVolumePairFromId(ImageId: integer; vInput, vOutput: TNNetVolume; ThreadDangerous: boolean = True);
      procedure GetRandomImagePair(vInput, vOutput: TNNetVolume);
      function ThreadSafeLoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume):boolean;

      property ClassCount: integer read FClassCount write FClassCount;
  end;

  { TClassesAndElements }
  TClassesAndElements = class(TStringStringListVolume)
    private
      FImageSubFolder: string;
      FBaseFolder: string;
      FNewSizeX, FNewSizeY: integer;
      FColorEncoding: integer;
      {$IFDEF HASTHREADS}FCritSecLoad: TRTLCriticalSection;{$ENDIF}
    public
      constructor Create();
      destructor Destroy(); override;
      function CountElements(): integer;
      procedure LoadFoldersAsClasses(FolderName: string; pImageSubFolder: string = ''; SkipFirst: integer = 0; SkipLast: integer = 0);
      procedure LoadFoldersAsClassesProportional(FolderName: string; pImageSubFolder: string; fSkipFirst: TNeuralFloat; fLoadLen: TNeuralFloat);
      procedure LoadImages(color_encoding: integer; NewSizeX: integer = 0; NewSizeY: integer = 0); overload;
      procedure LoadClass_FilenameFromFolder(FolderName: string);
      function GetRandomClassId(): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetClassesCount(): integer; {$IFDEF Release} inline; {$ENDIF}
      procedure GetRandomFileId(out ClassId:integer; out FileId:integer; StartPos: TNeuralFloat=0; Range: TNeuralFloat=1);
      procedure GetRandomFileName(out ClassId:integer; out FileName:string; StartPos: TNeuralFloat=0; Range: TNeuralFloat=1);
      procedure GetRandomImgVolumes(vInput, vOutput: TNNetVolume; StartPos: TNeuralFloat=0; Range: TNeuralFloat=1);
      function GetFileName(ClassId, ElementId: integer): string; {$IFDEF Release} inline; {$ENDIF}
      procedure AddVolumesTo(Volumes: TNNetVolumeList; EmptySource:boolean = false);
      procedure AddFileNamesTo(FileNames: TFileNameList);
      procedure MakeMonopolar(Divisor: TNeuralFloat = 4);
      function FileCountAtClassId(ClassId: integer): integer; {$IFDEF Release} inline; {$ENDIF}
      procedure LoadImages_NTL(index, threadnum: integer);
  end;

  /// add volumes into ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes
  // according to percentages found in TrainingPct, ValidationPct, TestPct.
  // folder names are classes
  procedure CreateVolumesFromImagesFromFolder(
    out ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    FolderName, pImageSubFolder: string;
    color_encoding: integer;
    TrainingProp, ValidationProp, TestProp: single;
    NewSizeX: integer = 0; NewSizeY: integer = 0);

  /// add file names into TrainingFileNames, ValidationFileNames, TestFileNames
  // according to percentages found in TrainingPct, ValidationPct, TestPct.
  // folder names are classes
  procedure CreateFileNameListsFromImagesFromFolder(
    out TrainingFileNames, ValidationFileNames, TestFileNames: TFileNameList;
    FolderName, pImageSubFolder: string;
    TrainingProp, ValidationProp, TestProp: single);

  {$IFDEF FPC}
  procedure LoadImageIntoVolume(M: TFPMemoryImage; Vol:TNNetVolume);
  procedure LoadVolumeIntoImage(Vol:TNNetVolume; M: TFPMemoryImage);
  function SaveImageFromVolumeIntoFile(V:TNNetVolume; ImageFileName:string):boolean;
  {$ENDIF}

  // Loads an image from a file and stores it into a Volume.
  function LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume):boolean; overload;

  // Loads an image from a file and stores it into a Volume resizing to
  // SizeX, SizeY and optionally encoding as neuronal input if has a
  // color encoding such as csEncodeRGB.
  function LoadImageFromFileIntoVolume(
    ImageFileName:string; V:TNNetVolume; SizeX, SizeY: integer;
    EncodeNeuronalInput: integer = -1):boolean; overload;

// Writes the header of a confusion matrix into a CSV file
procedure ConfusionWriteCSVHeader(var CSVConfusion: TextFile; Labels: array of string);

// Writes a confusion matrix into CSV file
procedure ConfusionWriteCSV(var CSVConfusion: TextFile; Vol: TNNetVolume; Digits: integer);

// Loads a TinyImage into TNNetVolume.
procedure LoadTinyImageIntoNNetVolume(var TI: TTinyImage; Vol: TNNetVolume); overload;
procedure LoadTinyImageIntoNNetVolume(var TI: TCifar100Image; Vol: TNNetVolume); overload;
procedure LoadTinyImageIntoNNetVolume(var TI: TMNistImage; Vol: TNNetVolume); overload;

// Loads a volume into a tiny image.
procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TTinyImage); overload;
procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TCifar100Image); overload;

// Loads a SingleChannelTinyImage Into TNNetVolue.
procedure LoadTinySingleChannelIntoNNetVolume(var SC: TTinySingleChannelImage; Vol: TNNetVolume);

// Creates a gray scale tiny image
procedure TinyImageCreateGrey(var TI: TTinyImage; var TIGrey: TTinySingleChannelImage);

// Calculates Horizontal Edges
procedure TinyImageHE(var TI, TIHE: TTinySingleChannelImage);

// Calculates Vertical Edges
procedure TinyImageVE(var TI, TIVE: TTinySingleChannelImage);

//Zeroes all pixels that have a small distance to the number 128
procedure TinyImageRemoveZeroGradient(var TI: TTinySingleChannelImage; distance: byte);

// Calculates Horizontal and Vertical Edges
procedure TinyImageHVE(var TI, TIHE: TTinySingleChannelImage);

// This function transforms a 2D TinyImage into 1D TinyImage
function TinyImageTo1D(var TI: TTinySingleChannelImage): TTinySingleChannelImage1D;

// creates CIFAR10 volumes required for training, testing and validation
procedure CreateCifar10Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  ValidationSampleSize: integer = 2000);

// creates CIFAR100 volumes required for training, testing and validation
procedure CreateCifar100Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  Verbose:boolean = true; ValidationSampleSize: integer = 2000);

// creates MNIST volumes required for training, testing and validation
procedure CreateMNISTVolumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList;
  TrainFileName, TestFileName: string;
  Verbose:boolean = true;
  IsFashion:boolean = false);

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; idx:integer; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; fileName:string; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;

// loads a CIFAR100 into TNNetVolumeList
procedure loadCifar100Dataset(ImgVolumes: TNNetVolumeList; fileName:string;
  color_encoding: byte = csEncodeRGB; Verbose:boolean = true);

// loads MNIST pair of files (image and labels) into ImgVolumes.
procedure loadMNISTDataset(ImgVolumes: TNNetVolumeList; fileName:string;
  Verbose:boolean = true; IsFashion:boolean = false;
  MaxLabel: integer = 10); overload;

// These functions return TRUE if the dataset is found or an error message otherwise
function CheckCIFARFile():boolean;
function CheckCIFAR100File():boolean;
function CheckMNISTFile(fileName:string; IsFasion:boolean = false):boolean;

// This function tests a neural network on the passed ImgVolumes
procedure TestBatch
(
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
);

// This function translates the original CIFAR10 labels to Animal/Machine labels.
procedure TranslateCifar10VolumesToMachineAnimal(VolumeList: TNNetVolumeList);

{
  RandomSubstring:
  This NLP function takes a string as input and returns a substring that starts
  immediately after a randomly selected space character within the input string.
  If there are no spaces in the input string, the entire string is returned as is.
  The function is useful for obtaining a random piece of text from a given string,
  which can be applied in various scenarios that require text randomization.

  Space positions are tracked using a TIntegerList. The Copy function is used
  to extract the substring from the randomly selected space position to the end
  of the input string.
}
function RandomSubstring(const InputString: string): string;

{
  RemoveRandomChars:
  This function takes a string and an integer count as input. It removes Count
  number of characters at random positions from the given string Str. The length
  of the string is recalculated in each iteration to account for the reduction in
  the string's length after each character removal.
}
function RemoveRandomChars(const Str: string; Count: integer): string;


// This function randomly removes one word from the input string.
function RemoveRandomWord(const Str: string): string;

type TNNetAAInteger = array of array of integer;

procedure FilterCSVWithNumbersUpToMax(inputFile,outputFile: string;
  MaxInteger: integer; MaxRows: integer = 0);

procedure LoadIntegersInCSV(filename: string;
  var aTokens: TNNetAAInteger; MaxRows: integer = 0);

{$IFNDEF FPC}
function SwapEndian(I:integer):integer;
procedure FindAllDirectories(AList: TStrings; const SearchPath: String;
  SearchSubDirs: Boolean = true; PathSeparator: char = ';'); overload;
function DirectorySeparator: string;
{$ENDIF}

// Simple character based NLP function for building a string from characters.
function GenerateStringFromChars(NN: TNNet; InputString: string; oSampler: TNNetSamplerBase = nil): string; overload;

// Takes a neural network (NN) and an input string, and returns the predicted class as an integer.
function GetClassFromChars(NN: TNNet; InputString: string): integer;

// Simple token based NLP function for building a string from an array of tokens.
function GenerateStringFromTokens(NN: TNNet; Dict:TStringListInt; InputString: string; oSampler: TNNetSamplerBase = nil): string;

function GenerateStringFromCasualCharNN(NN: TNNet;
  InputString: string; oSampler: TNNetSamplerBase = nil;
  EncodingMethod: integer = csNeuralEncodingMethodInt; EncodingMethod2: integer = 0): string;

// Re-encodes the whole prefix per token; for the streamed KV-cache/SSM-state sibling see GenerateTokensStreamed/GenerateStringStreamed in neuraldecode.
function GenerateStringFromCasualNN(NN: TNNet; Dict:TStringListInt;
  InputString: string; oSampler: TNNetSamplerBase = nil;
  EncodingMethod: integer = csNeuralEncodingMethodInt; EncodingMethod2: integer = 0): string;

// Simple function for debugging an NLP NN predicting the next token.
procedure DebugNLPOnPos(NN: TNNet; Dict: TStringListInt; var Dataset: TNNetAAInteger; Pos, Samples: integer);

// Converts a string into an array of integer
function StringToArrayOfInteger(InputString: string): TNeuralIntegerArray;

type
  // How TNNetSequencePacker lays documents out into fixed-length windows:
  // - pmSplitAcrossWindows: GPT-2/GPT-3 style packing. All documents are
  //   concatenated into one token stream (each document followed by one
  //   separator token) and the stream is cut into consecutive ContextLen
  //   windows; a document may be split across a window boundary. Only the
  //   final partial window is padded.
  // - pmNoSplitGreedy: greedy bin fill that never splits a document. Each
  //   document (+ its separator) is appended to the current window when it
  //   fits; otherwise the current window is padded and a new one starts.
  //   Documents longer than ContextLen-1 tokens are truncated.
  // - pmOneDocPerWindow: classic padded feeding (the baseline packing is
  //   compared against): one document per window, truncated to ContextLen-1
  //   tokens, followed by one separator and padding.
  TNNetPackingMode = (pmSplitAcrossWindows, pmNoSplitGreedy, pmOneDocPerWindow);

  { TNNetSequencePacker }
  // Sequence packing for autoregressive language-model pretraining: packs a
  // tokenized corpus (variable-length documents) into fixed ContextLen
  // training windows instead of padding each document, which is a large
  // throughput win on pad-heavy corpora.
  //
  // Conventions (matching the NLP pipeline in this repo, where token ids < 2
  // are special): pad = 0, end-of-document separator = 1, real corpus tokens
  // must be >= 2 (AddDocument raises EArgumentException otherwise). Every
  // document is followed by exactly one separator, so separator count equals
  // document count. The separator IS a training target (the model learns to
  // end documents); only positions whose next-token target is the pad token
  // are excluded from the loss (IsTargetPredictable = false). Within a
  // window, the target for position P is the token at position P+1, so the
  // last position of every window never has a target.
  //
  // Cross-document attention masking is intentionally NOT implemented:
  // TNNetScaledDotProductAttention only supports a static causal flag (plus
  // a static sliding window), not per-sample dynamic masks, so attention may
  // cross document boundaries inside a packed window. This is the standard
  // GPT-2/GPT-3 packing behaviour and works fine in practice.
  //
  // Typical use (per-position causal LM with a TNNetPointwiseSoftMax head):
  //   Packer.AddDocument(...); ...; Packer.Pack();
  //   Packer.GetTrainingPair(W, InputV, TargetV);
  //   NN.Compute(InputV);
  //   Packer.ApplyLossMask(W, TargetV, NN.GetLastLayer().Output);
  //   NN.Backpropagate(TargetV); // error = Output - Desired = 0 when masked
  // Coded by Claude (AI).
  TNNetSequencePacker = class(TObject)
    private
      FContextLen: integer;
      FMode: TNNetPackingMode;
      FSeparatorToken: integer;
      FPadToken: integer;
      FDocs: array of TNeuralIntegerArray;
      FDocCount: integer;
      FWindows: TNNetAAInteger;
      FIsPacked: boolean;
      procedure RequirePacked();
      procedure PackSplit();
      procedure PackGreedyBins(OneDocPerWindow: boolean);
    public
      constructor Create(pContextLen: integer;
        pMode: TNNetPackingMode = pmSplitAcrossWindows;
        pSeparatorToken: integer = 1; pPadToken: integer = 0);
      destructor Destroy; override;
      // Removes all documents and packed windows.
      procedure Clear();
      // Adds one tokenized document (without trailing separator; the packer
      // appends the separator itself). All token ids must be >= 2.
      procedure AddDocument(const Tokens: array of integer);
      // Char-level convenience: document tokens are Ord() of each character.
      procedure AddDocumentFromString(const Str: string);
      // Builds the packed windows from the documents added so far.
      procedure Pack();
      function WindowCount(): integer;
      // Returns a copy of window WindowIdx (always ContextLen tokens).
      function GetWindow(WindowIdx: integer): TNeuralIntegerArray;
      function GetToken(WindowIdx, Pos: integer): integer;
      // True when position Pos of window WindowIdx carries a loss: the target
      // (next token in the window) exists and is not the pad token.
      function IsTargetPredictable(WindowIdx, Pos: integer): boolean;
      function PredictableTargetCount(WindowIdx: integer): integer;
      // Fraction of loss-bearing target slots over all windows:
      // sum(PredictableTargetCount) / (WindowCount * (ContextLen-1)).
      function Utilization(): TNeuralFloat;
      // Fills a training pair for window WindowIdx. Input: token ids on the X
      // axis when pInput.Depth = 1 (embedding pipelines), one-hot across the
      // depth axis otherwise. Target: per-position one-hot of the next token
      // (rows without a predictable target are left all-zero - call
      // ApplyLossMask after Compute so they carry no gradient).
      procedure GetTrainingPair(WindowIdx: integer; pInput, pTarget: TNNetVolume);
      // Copies the actual network output into Desired at every position that
      // is not predictable. With the framework's error convention
      // e = Output - Desired this makes the error at masked positions exactly
      // zero, so no loss is backpropagated from pad targets.
      procedure ApplyLossMask(WindowIdx: integer; Desired, Actual: TNNetVolume);
      property ContextLen: integer read FContextLen;
      property Mode: TNNetPackingMode read FMode;
      property SeparatorToken: integer read FSeparatorToken;
      property PadToken: integer read FPadToken;
      property DocumentCount: integer read FDocCount;
  end;

const
  // Ignore-label convention for masked-LM training: a label of -1 means
  // "this position is NOT a loss target" (BERT/HuggingFace use -100, but this
  // repo's token ids are non-negative and any negative sentinel works; -1 is
  // used throughout TNNetMaskedLMCollator). Use BuildTrainingPair to turn the
  // integer label array into a one-hot target volume with all-zero rows at the
  // ignored positions (then ApplyLossMask after Compute, exactly like
  // TNNetSequencePacker, so e = Output - Desired is zero off the masked set).
  csMaskedLMIgnoreLabel = -1;

type
  { TNNetMaskedLMCollator }
  // BERT-style dynamic masked-language-model collator (a port of the
  // HuggingFace transformers DataCollatorForLanguageModeling, MLM mode).
  // Given a batch of token-id sequences, it independently corrupts each
  // sequence: a fraction MaskProb (default 0.15) of the non-special tokens is
  // selected for prediction; of the selected tokens 80% are replaced with the
  // [MASK] id, 10% with a uniformly random real token, and 10% are left
  // unchanged. Only the selected positions carry a loss; every other position
  // gets the ignore label csMaskedLMIgnoreLabel in the labels array, and the
  // selected positions carry the ORIGINAL (pre-corruption) token id.
  //
  // This unlocks encoder pretraining on top of the existing
  // AddTransformerEncoderBlock with no new layer types: feed CorruptedIds to
  // the encoder (token ids on the X axis when the input Depth = 1, one-hot on
  // the depth axis otherwise) and train a TNNetPointwiseSoftMax head against
  // the one-hot targets built by BuildTrainingPair; call ApplyLossMask after
  // Compute so the ignored positions backpropagate no gradient.
  //
  // Special tokens (pad/cls/sep/mask and anything passed to
  // AddSpecialTokenId) are NEVER selected for masking and never used as the
  // random replacement token.
  //
  // The masking RNG is seeded via Reseed so tests / runs are reproducible;
  // it uses an internal LCG independent of the global RandSeed so collation
  // does not perturb (or get perturbed by) weight-init randomness.
  // Coded by Claude (AI).
  TNNetMaskedLMCollator = class(TObject)
    private
      FMaskProb: TNeuralFloat;
      FReplaceMaskProb: TNeuralFloat;  // P(replace with [MASK]) within selected
      FRandomTokenProb: TNeuralFloat;  // P(replace with random token) within selected
      FMaskTokenId: integer;
      FVocabSize: integer;
      FSpecials: TNeuralIntegerArray;
      FSpecialCount: integer;
      FRngState: cardinal;
      function NextRandom(): TNeuralFloat;          // uniform [0,1)
      function NextRandomInt(N: integer): integer;  // uniform 0..N-1
      function IsSpecial(TokenId: integer): boolean;
      function RandomRealToken(): integer;          // a real (non-special) id
    public
      // pMaskTokenId: the [MASK] token id. pVocabSize: number of token ids
      // (random replacement draws from 0..pVocabSize-1, skipping specials).
      // The pad/cls/sep/mask ids and any AddSpecialTokenId id are never masked.
      constructor Create(pMaskTokenId, pVocabSize: integer;
        pMaskProb: TNeuralFloat = 0.15);
      destructor Destroy; override;
      // Registers a token id that must never be masked nor used as a random
      // replacement (the mask id itself is registered automatically).
      procedure AddSpecialTokenId(TokenId: integer);
      // Reseeds the internal RNG for reproducible masking.
      procedure Reseed(Seed: cardinal);
      // Corrupts one sequence in place of a copy. CorruptedIds receives the
      // masked input ids (same length as Tokens); Labels receives the original
      // id at each selected position and csMaskedLMIgnoreLabel everywhere else.
      procedure Collate(const Tokens: array of integer;
        out CorruptedIds, Labels: TNeuralIntegerArray);
      // Whole-word masking variant (a port of HuggingFace transformers
      // DataCollatorForWholeWordMask). WordIds is a per-token grouping array
      // parallel to Tokens: tokens carrying the SAME WordIds value are the
      // subword pieces of one word and the masking DECISION is taken once per
      // word -- if a word is selected, all of its pieces become loss targets
      // together (no partial-word masking). Within a selected word the existing
      // 80/10/10 mask/random/keep policy is applied INDEPENDENTLY per piece,
      // exactly as the HF reference does. A piece whose token is special is
      // never selected and never anchors a word (its WordIds value is ignored
      // for grouping of real pieces). Use any monotone-or-not integer scheme to
      // tag words; equal adjacent-or-not values are grouped. A common caller
      // convention is to give every special token its own unique WordIds value
      // (e.g. a decreasing counter) so specials never join a real word.
      procedure CollateWholeWord(const Tokens, WordIds: array of integer;
        out CorruptedIds, Labels: TNeuralIntegerArray);
      // Fills network volumes from one already-collated pair. pInput: corrupted
      // token ids on the X axis when pInput.Depth = 1 (embedding pipelines),
      // one-hot across the depth axis otherwise. pTarget: per-position one-hot
      // of the ORIGINAL token at selected positions, all-zero rows elsewhere
      // (so a softmax head with ApplyLossMask trains only on the masked set).
      procedure BuildTrainingPair(const CorruptedIds, Labels: TNeuralIntegerArray;
        pInput, pTarget: TNNetVolume);
      // Copies the network output into Desired at every ignored position so
      // that with e = Output - Desired the error there is exactly zero (same
      // contract as TNNetSequencePacker.ApplyLossMask). SeqLen rows are read
      // from Labels.
      procedure ApplyLossMask(const Labels: TNeuralIntegerArray;
        Desired, Actual: TNNetVolume);
      property MaskProb: TNeuralFloat read FMaskProb write FMaskProb;
      // The 80/10/10 split. ReplaceMaskProb + RandomTokenProb must be <= 1;
      // the remainder is the "leave unchanged" probability.
      property ReplaceMaskProb: TNeuralFloat read FReplaceMaskProb write FReplaceMaskProb;
      property RandomTokenProb: TNeuralFloat read FRandomTokenProb write FRandomTokenProb;
      property MaskTokenId: integer read FMaskTokenId;
      property VocabSize: integer read FVocabSize;
  end;

  { TNNetSpanCorruptionCollator }
  // T5 / SpanBERT-style span-corruption collator (a port of the T5
  // span_corruption objective, the BERT-relative sibling of
  // TNNetMaskedLMCollator). Instead of masking individual tokens in place it
  // masks CONTIGUOUS SPANS and produces a reshaped encoder-decoder pair:
  //
  //   * The INPUT is the original sequence with each masked span collapsed to a
  //     single unique sentinel token. Sentinels descend from a configurable
  //     base id: <extra_id_0> = SentinelBaseId, <extra_id_1> = SentinelBaseId-1,
  //     ... (the T5 convention puts <extra_id_0> at the top of the vocabulary).
  //   * The TARGET is the sentinel/span stream
  //       sentinel_0, span0_tokens, sentinel_1, span1_tokens, ..., final_sentinel
  //     i.e. each dropped span prefixed by the sentinel that replaced it, with a
  //     trailing sentinel (the T5 end marker) after the last span.
  //
  // Spans are sampled so that ~CorruptionRate of the (non-special) tokens are
  // masked, with span lengths drawn around MeanSpanLength (T5 defaults: 0.15
  // rate, mean length 3). Special tokens (anything passed to AddSpecialTokenId)
  // are never masked and break spans. Because the target is a RESHAPED, shorter
  // sequence than the input, this is a sibling class rather than a flag on
  // TNNetMaskedLMCollator. The (corrupted input, target) pair is exactly
  // round-trippable: the original sequence can be rebuilt by walking the input
  // and substituting, at each sentinel, the span that follows the matching
  // sentinel in the target.
  //
  // The RNG is an internal LCG seeded via Reseed, identical to and independent
  // of TNNetMaskedLMCollator's, so collation is reproducible and never perturbs
  // weight-init randomness.
  // Coded by Claude (AI).
  TNNetSpanCorruptionCollator = class(TObject)
    private
      FCorruptionRate: TNeuralFloat;
      FMeanSpanLength: TNeuralFloat;
      FSentinelBaseId: integer;
      FVocabSize: integer;
      FSpecials: TNeuralIntegerArray;
      FSpecialCount: integer;
      FRngState: cardinal;
      function NextRandom(): TNeuralFloat;
      function NextRandomInt(N: integer): integer;
      function IsSpecial(TokenId: integer): boolean;
      function SampleSpanLength(): integer;
    public
      // pSentinelBaseId: id of <extra_id_0> (sentinels count DOWN from here).
      // pVocabSize: number of token ids (sentinels are assumed to live at the
      // top of the vocabulary, so SentinelBaseId is typically VocabSize-1).
      // pCorruptionRate: target fraction of non-special tokens to mask.
      // pMeanSpanLength: mean masked-span length (>= 1).
      constructor Create(pSentinelBaseId, pVocabSize: integer;
        pCorruptionRate: TNeuralFloat = 0.15;
        pMeanSpanLength: TNeuralFloat = 3.0);
      destructor Destroy; override;
      // Registers a token id that must never be masked and that breaks spans.
      procedure AddSpecialTokenId(TokenId: integer);
      // Reseeds the internal RNG for reproducible span sampling.
      procedure Reseed(Seed: cardinal);
      // Returns the id of the I-th sentinel (<extra_id_I>): SentinelBaseId - I.
      function SentinelId(I: integer): integer;
      // Corrupts one sequence into a T5 (input, target) pair. SourceIds receives
      // the original sequence with each masked span replaced by one descending
      // sentinel; TargetIds receives the sentinel/span stream terminated by the
      // next (final) sentinel. NumSpans returns how many spans were masked.
      procedure Collate(const Tokens: array of integer;
        out SourceIds, TargetIds: TNeuralIntegerArray; out NumSpans: integer);
      // Fills network volumes from an already-collated pair (token ids on the X
      // axis when Depth = 1, one-hot on the depth axis otherwise). The encoder
      // is fed pSource; an autoregressive decoder is trained on pTarget.
      procedure BuildTrainingPair(const SourceIds, TargetIds: TNeuralIntegerArray;
        pSource, pTarget: TNNetVolume);
      property CorruptionRate: TNeuralFloat read FCorruptionRate write FCorruptionRate;
      property MeanSpanLength: TNeuralFloat read FMeanSpanLength write FMeanSpanLength;
      property SentinelBaseId: integer read FSentinelBaseId;
      property VocabSize: integer read FVocabSize;
  end;

implementation

uses
  math, neuralthread,
  {$IFDEF FPC}SysUtils,fileutil{$ELSE}
  SysUtils,
  IOUtils,
  Types
  {$ENDIF};

{$IFNDEF FPC}
function SwapEndian(I:integer):integer;
begin
  // valid for SmallInt
  // result := Swap(I)
  Result := ((Swap(Smallint(I)) and $ffff) shl $10) or (Swap(Smallint(I shr $10)) and $ffff)
end;

procedure FindAllDirectories(AList: TStrings; const SearchPath: String;
  SearchSubDirs: Boolean = true; PathSeparator: char = ';');
var
  dirs: TStringDynArray;
  dir, Path, SearchPattern: ShortString;
  SearchOption: TSearchOption;
begin
  if SearchSubDirs
  then SearchOption := TSearchOption.soAllDirectories
  else SearchOption := TSearchOption.soTopDirectoryOnly;
  Path := SearchPath;
  SearchPattern := '*';
  dirs := TDirectory.GetDirectories(Path, SearchPattern, SearchOption);//, SearchSubDirs);
  for dir in dirs do
  begin
    AList.Add(dir);
  end;
end;

procedure FindAllFiles(AList: TStrings; const SearchPath: String;
  const SearchMask: String = ''; SearchSubDirs: Boolean = True; DirAttr: Word = faDirectory;
  MaskSeparator: char = ';'; PathSeparator: char = ';');
var
  fileNames: TStringDynArray;
  fileName, Path, SearchPattern: ShortString;
  SearchOption: TSearchOption;
begin
  if SearchSubDirs
  then SearchOption := TSearchOption.soAllDirectories
  else SearchOption := TSearchOption.soTopDirectoryOnly;
  Path := SearchPath;
  SearchPattern := '*';
  fileNames := TDirectory.GetFiles(Path, SearchPattern, SearchOption);//, SearchSubDirs);
  for fileName in fileNames do
  begin
    AList.Add(fileName);
  end;
end;


function DirectorySeparator: string;
begin
  Result := TPath.DirectorySeparatorChar;
end;
{$ENDIF}

function GenerateStringFromChars(NN: TNNet; InputString: string;
  oSampler: TNNetSamplerBase): string;
var
  InputVolume, OutputVolume: TNNetVolume;
  NextTokenInt: integer;
  NextTokenChar: char;
  AB: array [0..0] of byte;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  repeat
    InputVolume.OneHotEncodingReversed(InputString);
    NN.Compute(InputVolume, OutputVolume);
    if (OutputVolume.Size = 8) then
    begin
      OutputVolume.ReadAsBits(AB, 0.5);
      NextTokenInt := AB[0];
    end
    else
    begin
      if Assigned(oSampler)
      then NextTokenInt := oSampler.GetToken(OutputVolume)
      else NextTokenInt := OutputVolume.GetClass();
    end;
    NextTokenChar := Char(NextTokenInt);
    if NextTokenInt > 1 then InputString := InputString + NextTokenChar;
  until (NextTokenInt < 2) or (Length(InputString)>=InputVolume.SizeX);
  Result := InputString;
  InputVolume.Free;
  OutputVolume.Free;
end;

// Takes a neural network (NN) and an input string,
// and returns the predicted class as an integer.
function GetClassFromChars(NN: TNNet; InputString: string): integer;
var
  InputVolume: TNNetVolume; // Declare a variable for the input volume.
begin
  // Create a new TNNetVolume based on the output size of the first layer of the neural network.
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);

  // Convert the input string into a one-hot encoded volume, which is the format
  // expected by the neural network for processing.
  InputVolume.OneHotEncodingReversed(InputString);

  // Run the forward pass of the neural network with the one-hot encoded input.
  NN.Compute(InputVolume);

  // After the network has computed the output, retrieve the class with the highest
  // probability from the last layer's output.
  Result := NN.GetLastLayer().Output.GetClass();

  // Release the memory allocated for the input volume to prevent memory leaks.
  InputVolume.Free;
end;

function GenerateStringFromTokens(NN: TNNet; Dict: TStringListInt;
  InputString: string; oSampler: TNNetSamplerBase): string;
var
  InputVolume, OutputVolume: TNNetVolume;
  NextTokenInt: integer;
  NextTokenStr: string;
  Tokens: TNeuralIntegerArray;
  TokenCnt: integer;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  Result := InputString;
  Dict.StringToIntegerArray(InputString, Tokens);
  TokenCnt := Length(Tokens);
  repeat
    InputVolume.CopyReversedNoChecksIntArr(Tokens);
    NN.Compute(InputVolume, OutputVolume);
    if Assigned(oSampler)
    then NextTokenInt := oSampler.GetToken(OutputVolume)
    else NextTokenInt := OutputVolume.GetClass();
    if NextTokenInt < Dict.Count then
    begin
      NextTokenStr := Dict.IntegerToWord(NextTokenInt);
      Result := Result + ' ' + NextTokenStr;
    end;
    TokenCnt := TokenCnt + 1;
    SetLength(Tokens, TokenCnt);
    Tokens[TokenCnt - 1] := NextTokenInt;
  until (NextTokenInt < 2) or (TokenCnt>=InputVolume.SizeX);
  SetLength(Tokens, 0);
  InputVolume.Free;
  OutputVolume.Free;
end;

function GenerateStringFromCasualCharNN(NN: TNNet; InputString: string;
  oSampler: TNNetSamplerBase; EncodingMethod: integer; EncodingMethod2: integer
  ): string;
var
  InputVolume, OutputVolume: TNNetVolume;
  NextTokenInt: integer;
  Tokens: TNeuralIntegerArray;
  TokenCnt: integer;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  if Length(InputString) > InputVolume.SizeX then
  begin
    InputString := Copy(InputString, Length(inputString) - InputVolume.SizeX + 1,InputVolume.SizeX);
  end;
  Result := InputString;
  Tokens := StringToArrayOfInteger(InputString);
  TokenCnt := Length(Tokens);
  repeat
    if      EncodingMethod = csNeuralEncodingMethodOneHot then InputVolume.OneHotEncoding(Tokens)
    else if EncodingMethod = csNeuralEncodingMethodGroupedOnHot then InputVolume.GroupedOneHotEncoding(Tokens, EncodingMethod2)
    else InputVolume.CopyNoChecksIntArr(Tokens);
    NN.Compute(InputVolume, OutputVolume);
    if Assigned(oSampler)
    then NextTokenInt := oSampler.GetTokenOnPixel(OutputVolume, TokenCnt-1, 0)
    else NextTokenInt := OutputVolume.GetClassOnPixel(TokenCnt - 1, 0);
    if NextTokenInt < 256 then
    begin
      Result := Result + Chr(NextTokenInt);
    end;
    TokenCnt := TokenCnt + 1;
    SetLength(Tokens, TokenCnt);
    Tokens[TokenCnt - 1] := NextTokenInt;
  until (NextTokenInt < 2) or (TokenCnt>=InputVolume.SizeX);
  SetLength(Tokens, 0);
  InputVolume.Free;
  OutputVolume.Free;
end;

function GenerateStringFromCasualNN(NN: TNNet; Dict: TStringListInt;
  InputString: string; oSampler: TNNetSamplerBase = nil;
  EncodingMethod: integer = csNeuralEncodingMethodInt; EncodingMethod2: integer = 0): string;
var
  InputVolume, OutputVolume: TNNetVolume;
  NextTokenInt: integer;
  NextTokenStr: string;
  Tokens: TNeuralIntegerArray;
  TokenCnt: integer;
  VocabCount: integer;
begin
  VocabCount := Dict.GetVocabCount();
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  Result := InputString;
  Dict.Tokenize(InputString, Tokens);
  TokenCnt := Length(Tokens);
  repeat
    if      EncodingMethod = csNeuralEncodingMethodOneHot then InputVolume.OneHotEncoding(Tokens)
    else if EncodingMethod = csNeuralEncodingMethodGroupedOnHot then InputVolume.GroupedOneHotEncoding(Tokens, EncodingMethod2)
    else InputVolume.CopyNoChecksIntArr(Tokens);
    NN.Compute(InputVolume, OutputVolume);
    if Assigned(oSampler)
    then NextTokenInt := oSampler.GetTokenOnPixel(OutputVolume, TokenCnt-1, 0)
    else NextTokenInt := OutputVolume.GetClassOnPixel(TokenCnt - 1, 0);
    if NextTokenInt < VocabCount then
    begin
      NextTokenStr := Dict.DeTokenize(NextTokenInt);
      // todo: make a more efficient code.
      if Dict.TokenizerHasSeparator
      then Result := Result + ' ' + NextTokenStr
      else Result := Result + NextTokenStr
    end;
    TokenCnt := TokenCnt + 1;
    SetLength(Tokens, TokenCnt);
    Tokens[TokenCnt - 1] := NextTokenInt;
  until (NextTokenInt < 2) or (TokenCnt>=InputVolume.SizeX);
  SetLength(Tokens, 0);
  InputVolume.Free;
  OutputVolume.Free;
end;

procedure DebugNLPOnPos(NN: TNNet; Dict: TStringListInt; var Dataset: TNNetAAInteger; Pos, Samples: integer);
var
  SampleId: integer;
  SampleLen: integer;
  SampleCutPosition: integer;
  ExpectedTokenInt: integer;
  AIntegerArray: TNeuralIntegerArray;
  pInput, pOutput: TNNetVolume;
  CntHit, CntMiss: integer;
  InputString: string;
begin
  pInput := TNNetVolume.Create();
  pOutput := TNNetVolume.Create();
  CntHit := 0;
  CntMiss := 0;
  // Make sure that expected input and output have the proper sizes.
  if NN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(NN.GetFirstLayer().Output);
  if NN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(NN.GetLastLayer().Output);
  for SampleId := 0 to Samples -1 do
  begin
    // Get the input sample
    SampleLen := Min(Length(Dataset[SampleId]), pInput.SizeX);
    SampleCutPosition := Min(SampleLen-1, Pos);
    // The expected token is the next character in the string
    ExpectedTokenInt := Dataset[SampleId][SampleCutPosition];
    // Encode the input and output volumes
    {$IFDEF FPC}
    AIntegerArray := Copy(Dataset[SampleId], 0, SampleCutPosition);
    {$ELSE}
    // This portion of code was coded by https://chatgpt.com/g/g-bqMxEDpIg-neural-api-free-pascal-developer
    SetLength(AIntegerArray, SampleCutPosition);
    if SampleCutPosition > 0 then
    Move(Dataset[SampleId][0], AIntegerArray[0], SampleCutPosition * SizeOf(Integer));
    {$ENDIF}
    pInput.Fill(0);
    pInput.CopyReversedNoChecksIntArr( AIntegerArray );
    NN.Compute(pInput, pOutput);
    if pOutput.GetClass() = ExpectedTokenInt then
    begin
      Inc(CntHit);
      if random(100) = 0 then
      begin
        InputString := Dict.IntegerArrayToString(AIntegerArray);
        WriteLn(InputString,'-->',Dict.IntegerToWord(ExpectedTokenInt));
        WriteLn(GenerateStringFromTokens(NN, Dict, InputString, nil),'.');
      end;
    end
    else
    begin
      Inc(CntMiss);
    end;
  end;
  WriteLn('Pos: ',Pos,' Hit:',CntHit);
  pOutput.Free;
  pInput.Free;
end;

function StringToArrayOfInteger(InputString: string): TNeuralIntegerArray;
var
  InputLen, CntChar: integer;
begin
  InputLen := Length(InputString);
  SetLength(Result, InputLen);

  if InputLen>0 then
  begin
    for CntChar := 1 to InputLen do
    begin
      Result[CntChar-1] := ord(InputString[CntChar]);
    end;
  end;
end;

{ TNNetSequencePacker }

constructor TNNetSequencePacker.Create(pContextLen: integer;
  pMode: TNNetPackingMode; pSeparatorToken: integer; pPadToken: integer);
begin
  inherited Create();
  if pContextLen < 2 then
    raise Exception.Create('TNNetSequencePacker: ContextLen must be >= 2.');
  if pSeparatorToken = pPadToken then
    raise Exception.Create(
      'TNNetSequencePacker: separator and pad tokens must differ.');
  FContextLen := pContextLen;
  FMode := pMode;
  FSeparatorToken := pSeparatorToken;
  FPadToken := pPadToken;
  Clear();
end;

destructor TNNetSequencePacker.Destroy;
begin
  Clear();
  inherited Destroy;
end;

procedure TNNetSequencePacker.Clear();
var
  I: integer;
begin
  for I := 0 to Length(FDocs) - 1 do SetLength(FDocs[I], 0);
  SetLength(FDocs, 0);
  FDocCount := 0;
  for I := 0 to Length(FWindows) - 1 do SetLength(FWindows[I], 0);
  SetLength(FWindows, 0);
  FIsPacked := false;
end;

procedure TNNetSequencePacker.AddDocument(const Tokens: array of integer);
var
  I: integer;
begin
  for I := 0 to Length(Tokens) - 1 do
  begin
    if Tokens[I] < 2 then
      raise Exception.Create(
        'TNNetSequencePacker: real token ids must be >= 2 ' +
        '(ids < 2 are reserved for pad/separator); got ' +
        IntToStr(Tokens[I]) + '.');
  end;
  if FDocCount >= Length(FDocs) then SetLength(FDocs, 8 + FDocCount * 2);
  SetLength(FDocs[FDocCount], Length(Tokens));
  for I := 0 to Length(Tokens) - 1 do FDocs[FDocCount][I] := Tokens[I];
  Inc(FDocCount);
  FIsPacked := false;
end;

procedure TNNetSequencePacker.AddDocumentFromString(const Str: string);
var
  Tokens: TNeuralIntegerArray;
begin
  Tokens := StringToArrayOfInteger(Str);
  AddDocument(Tokens);
  SetLength(Tokens, 0);
end;

procedure TNNetSequencePacker.RequirePacked();
begin
  if not FIsPacked then
    raise Exception.Create('TNNetSequencePacker: call Pack() first.');
end;

// GPT-style v1 packing: one continuous stream (doc + separator per document)
// cut into consecutive ContextLen windows; documents may be split across a
// window boundary; only the final partial window is padded.
procedure TNNetSequencePacker.PackSplit();
var
  StreamLen, DocIdx, I, Pos, WinCount, W: integer;
  Stream: TNeuralIntegerArray;
begin
  StreamLen := 0;
  for DocIdx := 0 to FDocCount - 1 do
    StreamLen := StreamLen + Length(FDocs[DocIdx]) + 1; // +1 separator
  SetLength(Stream, StreamLen);
  Pos := 0;
  for DocIdx := 0 to FDocCount - 1 do
  begin
    for I := 0 to Length(FDocs[DocIdx]) - 1 do
    begin
      Stream[Pos] := FDocs[DocIdx][I];
      Inc(Pos);
    end;
    Stream[Pos] := FSeparatorToken;
    Inc(Pos);
  end;
  WinCount := (StreamLen + FContextLen - 1) div FContextLen;
  SetLength(FWindows, WinCount);
  for W := 0 to WinCount - 1 do
  begin
    SetLength(FWindows[W], FContextLen);
    for I := 0 to FContextLen - 1 do
    begin
      Pos := W * FContextLen + I;
      if Pos < StreamLen
      then FWindows[W][I] := Stream[Pos]
      else FWindows[W][I] := FPadToken;
    end;
  end;
  SetLength(Stream, 0);
end;

// No-split greedy bin fill (and, when OneDocPerWindow, the padded baseline):
// a document (+ its separator) never crosses a window boundary; documents
// longer than ContextLen-1 tokens are truncated so doc+separator always fits.
procedure TNNetSequencePacker.PackGreedyBins(OneDocPerWindow: boolean);
var
  DocIdx, DocLen, I, Pos, W: integer;

  procedure PadAndCloseCurrent();
  var
    P: integer;
  begin
    for P := Pos to FContextLen - 1 do FWindows[W][P] := FPadToken;
    Pos := FContextLen;
  end;

  procedure OpenNewWindow();
  begin
    Inc(W);
    if W >= Length(FWindows) then SetLength(FWindows, 8 + W * 2);
    SetLength(FWindows[W], FContextLen);
    Pos := 0;
  end;

begin
  W := -1;
  Pos := FContextLen; // force a new window for the first document
  for DocIdx := 0 to FDocCount - 1 do
  begin
    DocLen := Length(FDocs[DocIdx]);
    if DocLen > FContextLen - 1 then DocLen := FContextLen - 1;
    if (Pos + DocLen + 1 > FContextLen) or
      (OneDocPerWindow and (Pos > 0)) then
    begin
      if W >= 0 then PadAndCloseCurrent();
      OpenNewWindow();
    end;
    for I := 0 to DocLen - 1 do
    begin
      FWindows[W][Pos] := FDocs[DocIdx][I];
      Inc(Pos);
    end;
    FWindows[W][Pos] := FSeparatorToken;
    Inc(Pos);
  end;
  if W >= 0 then PadAndCloseCurrent();
  SetLength(FWindows, W + 1);
end;

procedure TNNetSequencePacker.Pack();
begin
  case FMode of
    pmSplitAcrossWindows: PackSplit();
    pmNoSplitGreedy: PackGreedyBins(false);
    pmOneDocPerWindow: PackGreedyBins(true);
  end;
  FIsPacked := true;
end;

function TNNetSequencePacker.WindowCount(): integer;
begin
  RequirePacked();
  Result := Length(FWindows);
end;

function TNNetSequencePacker.GetWindow(WindowIdx: integer): TNeuralIntegerArray;
var
  I: integer;
begin
  RequirePacked();
  SetLength(Result, FContextLen);
  for I := 0 to FContextLen - 1 do Result[I] := FWindows[WindowIdx][I];
end;

function TNNetSequencePacker.GetToken(WindowIdx, Pos: integer): integer;
begin
  RequirePacked();
  Result := FWindows[WindowIdx][Pos];
end;

function TNNetSequencePacker.IsTargetPredictable(WindowIdx, Pos: integer): boolean;
begin
  RequirePacked();
  Result := (Pos >= 0) and (Pos < FContextLen - 1) and
    (FWindows[WindowIdx][Pos + 1] <> FPadToken);
end;

function TNNetSequencePacker.PredictableTargetCount(WindowIdx: integer): integer;
var
  Pos: integer;
begin
  Result := 0;
  for Pos := 0 to FContextLen - 2 do
    if IsTargetPredictable(WindowIdx, Pos) then Inc(Result);
end;

function TNNetSequencePacker.Utilization(): TNeuralFloat;
var
  W, TotalSlots, Predictable: integer;
begin
  RequirePacked();
  Result := 0;
  TotalSlots := Length(FWindows) * (FContextLen - 1);
  if TotalSlots = 0 then exit;
  Predictable := 0;
  for W := 0 to Length(FWindows) - 1 do
    Predictable := Predictable + PredictableTargetCount(W);
  Result := Predictable / TotalSlots;
end;

procedure TNNetSequencePacker.GetTrainingPair(WindowIdx: integer;
  pInput, pTarget: TNNetVolume);
var
  Pos, Token: integer;
begin
  RequirePacked();
  if pInput.SizeX <> FContextLen then
    raise Exception.Create('TNNetSequencePacker.GetTrainingPair: input SizeX ' +
      IntToStr(pInput.SizeX) + ' <> ContextLen ' + IntToStr(FContextLen) + '.');
  if pTarget.SizeX <> FContextLen then
    raise Exception.Create('TNNetSequencePacker.GetTrainingPair: target SizeX ' +
      IntToStr(pTarget.SizeX) + ' <> ContextLen ' + IntToStr(FContextLen) + '.');
  pInput.Fill(0);
  for Pos := 0 to FContextLen - 1 do
  begin
    Token := FWindows[WindowIdx][Pos];
    if pInput.Depth = 1
    then pInput[Pos, 0, 0] := Token              // token ids -> embedding
    else if Token < pInput.Depth
    then pInput[Pos, 0, Token] := 1;             // one-hot across depth
  end;
  pTarget.Fill(0);
  for Pos := 0 to FContextLen - 2 do
  begin
    if IsTargetPredictable(WindowIdx, Pos) then
    begin
      Token := FWindows[WindowIdx][Pos + 1];
      if Token < pTarget.Depth then pTarget[Pos, 0, Token] := 1;
    end;
  end;
end;

procedure TNNetSequencePacker.ApplyLossMask(WindowIdx: integer;
  Desired, Actual: TNNetVolume);
var
  Pos, D: integer;
begin
  RequirePacked();
  for Pos := 0 to FContextLen - 1 do
  begin
    if not IsTargetPredictable(WindowIdx, Pos) then
    begin
      for D := 0 to Desired.Depth - 1 do
        Desired[Pos, 0, D] := Actual[Pos, 0, D];
    end;
  end;
end;

{ TNNetMaskedLMCollator }

constructor TNNetMaskedLMCollator.Create(pMaskTokenId, pVocabSize: integer;
  pMaskProb: TNeuralFloat);
begin
  inherited Create();
  if pVocabSize < 2 then
    raise Exception.Create('TNNetMaskedLMCollator: VocabSize must be >= 2.');
  if (pMaskTokenId < 0) or (pMaskTokenId >= pVocabSize) then
    raise Exception.Create(
      'TNNetMaskedLMCollator: MaskTokenId must be in 0..VocabSize-1.');
  if (pMaskProb < 0) or (pMaskProb > 1) then
    raise Exception.Create(
      'TNNetMaskedLMCollator: MaskProb must be in [0,1].');
  FMaskTokenId := pMaskTokenId;
  FVocabSize := pVocabSize;
  FMaskProb := pMaskProb;
  FReplaceMaskProb := 0.8;
  FRandomTokenProb := 0.1;
  FSpecialCount := 0;
  SetLength(FSpecials, 0);
  FRngState := 305419896; // arbitrary non-zero default seed
  // The mask token is special: never selected, never a random replacement.
  AddSpecialTokenId(pMaskTokenId);
end;

destructor TNNetMaskedLMCollator.Destroy;
begin
  SetLength(FSpecials, 0);
  inherited Destroy;
end;

procedure TNNetMaskedLMCollator.AddSpecialTokenId(TokenId: integer);
begin
  if IsSpecial(TokenId) then exit;
  if FSpecialCount >= Length(FSpecials) then
    SetLength(FSpecials, 8 + FSpecialCount * 2);
  FSpecials[FSpecialCount] := TokenId;
  Inc(FSpecialCount);
end;

procedure TNNetMaskedLMCollator.Reseed(Seed: cardinal);
begin
  if Seed = 0 then Seed := 1; // a zero state would stick at zero
  FRngState := Seed;
end;

function TNNetMaskedLMCollator.NextRandom(): TNeuralFloat;
begin
  // Numerical Recipes LCG; the high bits are well-mixed, so use them for the
  // [0,1) draw. Independent of the global RandSeed.
  FRngState := FRngState * 1664525 + 1013904223;
  Result := (FRngState shr 8) / 16777216.0; // top 24 bits / 2^24
end;

function TNNetMaskedLMCollator.NextRandomInt(N: integer): integer;
begin
  if N <= 1 then begin Result := 0; exit; end;
  Result := Trunc(NextRandom() * N);
  if Result >= N then Result := N - 1; // guard the (vanishingly rare) edge
end;

function TNNetMaskedLMCollator.IsSpecial(TokenId: integer): boolean;
var
  I: integer;
begin
  Result := false;
  for I := 0 to FSpecialCount - 1 do
    if FSpecials[I] = TokenId then begin Result := true; exit; end;
end;

function TNNetMaskedLMCollator.RandomRealToken(): integer;
begin
  // Rejection-sample a non-special id in 0..VocabSize-1.
  repeat
    Result := NextRandomInt(FVocabSize);
  until not IsSpecial(Result);
end;

procedure TNNetMaskedLMCollator.Collate(const Tokens: array of integer;
  out CorruptedIds, Labels: TNeuralIntegerArray);
var
  Len, I: integer;
  Roll: TNeuralFloat;
begin
  Len := Length(Tokens);
  SetLength(CorruptedIds, Len);
  SetLength(Labels, Len);
  for I := 0 to Len - 1 do
  begin
    CorruptedIds[I] := Tokens[I];
    Labels[I] := csMaskedLMIgnoreLabel;
    if IsSpecial(Tokens[I]) then continue;
    if NextRandom() >= FMaskProb then continue; // not selected for prediction
    // Selected: this position carries the loss; remember the original id.
    Labels[I] := Tokens[I];
    Roll := NextRandom();
    if Roll < FReplaceMaskProb then
      CorruptedIds[I] := FMaskTokenId                    // 80%: [MASK]
    else if Roll < FReplaceMaskProb + FRandomTokenProb then
      CorruptedIds[I] := RandomRealToken()               // 10%: random token
    // else 10%: leave CorruptedIds[I] unchanged.
    ;
  end;
end;

procedure TNNetMaskedLMCollator.CollateWholeWord(
  const Tokens, WordIds: array of integer;
  out CorruptedIds, Labels: TNeuralIntegerArray);
var
  Len, I, J: integer;
  Roll: TNeuralFloat;
  WordSelected: boolean;
begin
  Len := Length(Tokens);
  if Length(WordIds) <> Len then
    raise Exception.Create(
      'TNNetMaskedLMCollator.CollateWholeWord: Tokens and WordIds lengths ' +
      'differ.');
  SetLength(CorruptedIds, Len);
  SetLength(Labels, Len);
  for I := 0 to Len - 1 do
  begin
    CorruptedIds[I] := Tokens[I];
    Labels[I] := csMaskedLMIgnoreLabel;
  end;
  // Walk contiguous runs of equal WordIds (one word = one run of pieces). The
  // mask/keep DECISION is taken once per word; special pieces are skipped and
  // break the run so they never join a real word.
  I := 0;
  while I < Len do
  begin
    if IsSpecial(Tokens[I]) then
    begin
      Inc(I);
      continue;
    end;
    // Extent of this word: pieces sharing WordIds[I], stopping at a special.
    J := I + 1;
    while (J < Len) and (WordIds[J] = WordIds[I]) and (not IsSpecial(Tokens[J])) do
      Inc(J);
    // One selection draw for the whole word [I, J).
    WordSelected := NextRandom() < FMaskProb;
    if WordSelected then
    begin
      while I < J do
      begin
        Labels[I] := Tokens[I];
        // HF applies the 80/10/10 split independently to each piece.
        Roll := NextRandom();
        if Roll < FReplaceMaskProb then
          CorruptedIds[I] := FMaskTokenId
        else if Roll < FReplaceMaskProb + FRandomTokenProb then
          CorruptedIds[I] := RandomRealToken()
        ;
        Inc(I);
      end;
    end
    else
      I := J;
  end;
end;

procedure TNNetMaskedLMCollator.BuildTrainingPair(
  const CorruptedIds, Labels: TNeuralIntegerArray; pInput, pTarget: TNNetVolume);
var
  Len, P: integer;
begin
  Len := Length(CorruptedIds);
  if Length(Labels) <> Len then
    raise Exception.Create(
      'TNNetMaskedLMCollator.BuildTrainingPair: CorruptedIds and Labels ' +
      'lengths differ.');
  if pInput.SizeX < Len then
    raise Exception.Create(
      'TNNetMaskedLMCollator.BuildTrainingPair: input SizeX (' +
      IntToStr(pInput.SizeX) + ') < sequence length (' + IntToStr(Len) + ').');
  if pTarget.SizeX < Len then
    raise Exception.Create(
      'TNNetMaskedLMCollator.BuildTrainingPair: target SizeX (' +
      IntToStr(pTarget.SizeX) + ') < sequence length (' + IntToStr(Len) + ').');
  pInput.Fill(0);
  pTarget.Fill(0);
  for P := 0 to Len - 1 do
  begin
    if pInput.Depth = 1 then
      pInput[P, 0, 0] := CorruptedIds[P]            // token ids on the X axis
    else
      pInput[P, 0, CorruptedIds[P]] := 1;           // one-hot on the depth axis
    // Target: one-hot of the ORIGINAL id only at selected positions; ignored
    // rows stay all-zero (no loss with ApplyLossMask).
    if Labels[P] <> csMaskedLMIgnoreLabel then
      pTarget[P, 0, Labels[P]] := 1;
  end;
end;

procedure TNNetMaskedLMCollator.ApplyLossMask(const Labels: TNeuralIntegerArray;
  Desired, Actual: TNNetVolume);
var
  P, D: integer;
begin
  for P := 0 to Length(Labels) - 1 do
    if Labels[P] = csMaskedLMIgnoreLabel then
      for D := 0 to Desired.Depth - 1 do
        Desired[P, 0, D] := Actual[P, 0, D];
end;

{ TNNetSpanCorruptionCollator }

constructor TNNetSpanCorruptionCollator.Create(
  pSentinelBaseId, pVocabSize: integer;
  pCorruptionRate, pMeanSpanLength: TNeuralFloat);
begin
  inherited Create();
  if pVocabSize < 2 then
    raise Exception.Create('TNNetSpanCorruptionCollator: VocabSize must be >= 2.');
  if (pSentinelBaseId < 0) or (pSentinelBaseId >= pVocabSize) then
    raise Exception.Create(
      'TNNetSpanCorruptionCollator: SentinelBaseId must be in 0..VocabSize-1.');
  if (pCorruptionRate < 0) or (pCorruptionRate > 1) then
    raise Exception.Create(
      'TNNetSpanCorruptionCollator: CorruptionRate must be in [0,1].');
  if pMeanSpanLength < 1 then
    raise Exception.Create(
      'TNNetSpanCorruptionCollator: MeanSpanLength must be >= 1.');
  FSentinelBaseId := pSentinelBaseId;
  FVocabSize := pVocabSize;
  FCorruptionRate := pCorruptionRate;
  FMeanSpanLength := pMeanSpanLength;
  FSpecialCount := 0;
  SetLength(FSpecials, 0);
  FRngState := 305419896;
end;

destructor TNNetSpanCorruptionCollator.Destroy;
begin
  SetLength(FSpecials, 0);
  inherited Destroy;
end;

procedure TNNetSpanCorruptionCollator.AddSpecialTokenId(TokenId: integer);
begin
  if IsSpecial(TokenId) then exit;
  if FSpecialCount >= Length(FSpecials) then
    SetLength(FSpecials, 8 + FSpecialCount * 2);
  FSpecials[FSpecialCount] := TokenId;
  Inc(FSpecialCount);
end;

procedure TNNetSpanCorruptionCollator.Reseed(Seed: cardinal);
begin
  if Seed = 0 then Seed := 1;
  FRngState := Seed;
end;

function TNNetSpanCorruptionCollator.NextRandom(): TNeuralFloat;
begin
  FRngState := FRngState * 1664525 + 1013904223;
  Result := (FRngState shr 8) / 16777216.0;
end;

function TNNetSpanCorruptionCollator.NextRandomInt(N: integer): integer;
begin
  if N <= 1 then begin Result := 0; exit; end;
  Result := Trunc(NextRandom() * N);
  if Result >= N then Result := N - 1;
end;

function TNNetSpanCorruptionCollator.IsSpecial(TokenId: integer): boolean;
var
  I: integer;
begin
  Result := false;
  for I := 0 to FSpecialCount - 1 do
    if FSpecials[I] = TokenId then begin Result := true; exit; end;
end;

function TNNetSpanCorruptionCollator.SampleSpanLength(): integer;
begin
  // Geometric-ish span length with the requested mean, clamped to >= 1. A
  // geometric distribution with success p = 1/mean has mean 1/p = MeanSpanLength.
  Result := 1;
  while (NextRandom() > (1.0 / FMeanSpanLength)) do
  begin
    Inc(Result);
    if Result >= 256 then break; // safety clamp
  end;
end;

function TNNetSpanCorruptionCollator.SentinelId(I: integer): integer;
begin
  Result := FSentinelBaseId - I;
end;

procedure TNNetSpanCorruptionCollator.Collate(const Tokens: array of integer;
  out SourceIds, TargetIds: TNeuralIntegerArray; out NumSpans: integer);
var
  Len, I, Budget, SpanLen, SrcLen, TgtLen, SpanEnd: integer;
  Masked: array of boolean;
begin
  Len := Length(Tokens);
  NumSpans := 0;
  SetLength(Masked, Len);
  for I := 0 to Len - 1 do Masked[I] := false;
  // Token budget to mask (~CorruptionRate of non-special tokens).
  Budget := 0;
  for I := 0 to Len - 1 do
    if not IsSpecial(Tokens[I]) then Inc(Budget);
  Budget := Round(Budget * FCorruptionRate);

  // Greedily sample spans left-to-right until the budget is spent. A coin per
  // start position keeps spans spread out; an accepted span is the next run of
  // SampleSpanLength() non-special tokens.
  I := 0;
  while (I < Len) and (Budget > 0) do
  begin
    if IsSpecial(Tokens[I]) then begin Inc(I); continue; end;
    // Probability of starting a span here, tuned so spans cover ~the budget.
    if NextRandom() < (1.0 / FMeanSpanLength) then
    begin
      SpanLen := SampleSpanLength();
      if SpanLen > Budget then SpanLen := Budget;
      SpanEnd := I;
      while (SpanEnd < Len) and (SpanLen > 0) and (not IsSpecial(Tokens[SpanEnd])) do
      begin
        Masked[SpanEnd] := true;
        Dec(Budget);
        Dec(SpanLen);
        Inc(SpanEnd);
      end;
      Inc(NumSpans);
      // Leave at least one unmasked token before the next span may start.
      I := SpanEnd + 1;
    end
    else
      Inc(I);
  end;

  // Build the source (sentinel-collapsed) and target (sentinel/span) streams.
  SetLength(SourceIds, Len);     // upper bound; trimmed below
  SetLength(TargetIds, Len + NumSpans + 1); // spans + leading/trailing sentinels
  SrcLen := 0;
  TgtLen := 0;
  NumSpans := 0;
  I := 0;
  while I < Len do
  begin
    if Masked[I] then
    begin
      // Emit one sentinel into source; open the span in target.
      SourceIds[SrcLen] := SentinelId(NumSpans); Inc(SrcLen);
      TargetIds[TgtLen] := SentinelId(NumSpans); Inc(TgtLen);
      while (I < Len) and Masked[I] do
      begin
        TargetIds[TgtLen] := Tokens[I]; Inc(TgtLen);
        Inc(I);
      end;
      Inc(NumSpans);
    end
    else
    begin
      SourceIds[SrcLen] := Tokens[I]; Inc(SrcLen);
      Inc(I);
    end;
  end;
  // Trailing (final) sentinel after the last span, per the T5 target format.
  TargetIds[TgtLen] := SentinelId(NumSpans); Inc(TgtLen);
  SetLength(SourceIds, SrcLen);
  SetLength(TargetIds, TgtLen);
end;

procedure TNNetSpanCorruptionCollator.BuildTrainingPair(
  const SourceIds, TargetIds: TNeuralIntegerArray; pSource, pTarget: TNNetVolume);
var
  P: integer;
begin
  if pSource.SizeX < Length(SourceIds) then
    raise Exception.Create(
      'TNNetSpanCorruptionCollator.BuildTrainingPair: source SizeX (' +
      IntToStr(pSource.SizeX) + ') < source length (' +
      IntToStr(Length(SourceIds)) + ').');
  if pTarget.SizeX < Length(TargetIds) then
    raise Exception.Create(
      'TNNetSpanCorruptionCollator.BuildTrainingPair: target SizeX (' +
      IntToStr(pTarget.SizeX) + ') < target length (' +
      IntToStr(Length(TargetIds)) + ').');
  pSource.Fill(0);
  pTarget.Fill(0);
  for P := 0 to Length(SourceIds) - 1 do
    if pSource.Depth = 1 then
      pSource[P, 0, 0] := SourceIds[P]
    else
      pSource[P, 0, SourceIds[P]] := 1;
  for P := 0 to Length(TargetIds) - 1 do
    if pTarget.Depth = 1 then
      pTarget[P, 0, 0] := TargetIds[P]
    else
      pTarget[P, 0, TargetIds[P]] := 1;
end;

procedure CreateVolumesFromImagesFromFolder(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList;
  FolderName, pImageSubFolder: string;
  color_encoding: integer;
  TrainingProp, ValidationProp, TestProp: single;
  NewSizeX: integer = 0; NewSizeY: integer = 0);
var
  ClassesAndElements: TClassesAndElements;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();

  ClassesAndElements := TClassesAndElements.Create();

  if ValidationProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, TrainingProp, ValidationProp);
    ClassesAndElements.LoadImages(color_encoding, NewSizeX, NewSizeY);
    ClassesAndElements.AddVolumesTo(ImgValidationVolumes, {EmptySource=}true);
    ClassesAndElements.Clear;
  end;

  if TestProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, TrainingProp + ValidationProp, TestProp);
    ClassesAndElements.LoadImages(color_encoding, NewSizeX, NewSizeY);
    ClassesAndElements.AddVolumesTo(ImgTestVolumes, {EmptySource=}true);
    ClassesAndElements.Clear;
  end;

  if TrainingProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, 0, TrainingProp);
    ClassesAndElements.LoadImages(color_encoding, NewSizeX, NewSizeY);
    ClassesAndElements.AddVolumesTo(ImgTrainingVolumes, {EmptySource=}true);
    ClassesAndElements.Clear;
  end;

  ClassesAndElements.Free;
end;

constructor TFileNameList.Create();
begin
  inherited Create();
  {$IFDEF HASTHREADS}
  NeuralInitCriticalSection(FCritSecLoad);
  {$ENDIF}
end;

destructor TFileNameList.Destroy();
begin
  {$IFDEF HASTHREADS}
  NeuralDoneCriticalSection(FCritSecLoad);
  {$ENDIF}
  inherited Destroy();
end;

{ TFileNameList }
procedure TFileNameList.GetImageVolumePairFromId(ImageId: integer; vInput, vOutput: TNNetVolume; ThreadDangerous: boolean = True);
var
  FileName: string;
  ClassId: integer;
begin
  FileName := Self[ImageId];
  ClassId := Self.Integers[Imageid];
  if ThreadDangerous
  then ThreadSafeLoadImageFromFileIntoVolume(FileName, vInput)
  else LoadImageFromFileIntoVolume(FileName, vInput);
  vInput.Tag := ClassId;
  vOutput.Tag := ClassId;
  vInput.Divi(64);
  vInput.Sub(2);
  vOutput.Resize(FClassCount);
  vOutput.SetClassForSoftMax(ClassId);
end;

procedure TFileNameList.GetRandomImagePair(vInput, vOutput: TNNetVolume);
begin
  GetImageVolumePairFromId(Self.GetRandomIndex(), vInput, vOutput);
end;

{ TClassesAndElements }
constructor TClassesAndElements.Create();
begin
  inherited Create();
  FImageSubFolder := '';
  FBaseFolder := '';
  {$IFDEF HASTHREADS}
  NeuralInitCriticalSection(FCritSecLoad);
  {$ENDIF}
end;

destructor TClassesAndElements.Destroy();
begin
  {$IFDEF HASTHREADS}
  NeuralDoneCriticalSection(FCritSecLoad);
  {$ENDIF}
  inherited Destroy();
end;

function TClassesAndElements.CountElements(): integer;
var
  ClassId: integer;
begin
  Result := 0;
  if Count > 0 then
  begin
    for ClassId := 0 to Count - 1 do
    begin
      Result := Result + Self.List[ClassId].Count;
    end;
  end;
end;

procedure TClassesAndElements.LoadFoldersAsClasses(FolderName: string; pImageSubFolder: string = ''; SkipFirst: integer = 0; SkipLast: integer = 0);
var
  ClassCnt: integer;
  MaxClass: integer;
  ClassFolder: string;
begin
  FBaseFolder := FolderName;
  FImageSubFolder := pImageSubFolder;
  FindAllDirectories(Self, FolderName, {SearchSubDirs}false);
  FixObjects();
  //WriteLn(FolderName,':',Self.Count);
  if Self.Count > 0 then
  begin
    MaxClass := Self.Count - 1;
    begin
      for ClassCnt := 0 to MaxClass do
      begin
        ClassFolder := Self[ClassCnt] + DirectorySeparator;
        if FImageSubFolder <> '' then
        begin
          ClassFolder := ClassFolder + FImageSubFolder + DirectorySeparator;
        end;
        if not Assigned(Self.List[ClassCnt]) then
        begin
          WriteLn(ClassFolder,' - error: not assigned list');
        end;
        FindAllFiles(Self.List[ClassCnt], ClassFolder, '*.png;*.jpg;*.jpeg;*.bmp', {SearchSubDirs} false);
        Self.List[ClassCnt].FixObjects();
        if SkipFirst > 0 then Self.List[ClassCnt].DeleteFirst(SkipFirst);
        if SkipLast > 0 then Self.List[ClassCnt].DeleteLast(SkipLast);
        //WriteLn(ClassFolder,':',Self.List[ClassCnt].Count);
      end;
    end;
  end;
end;

procedure TClassesAndElements.LoadFoldersAsClassesProportional(
  FolderName: string; pImageSubFolder: string; fSkipFirst: TNeuralFloat;
  fLoadLen: TNeuralFloat);
var
  ClassCnt, ElementCnt: integer;
  MaxClass, SkipFirst, SkipLast, Loading: integer;
  ClassFolder: string;
begin
  FBaseFolder := FolderName;
  FImageSubFolder := pImageSubFolder;
  FindAllDirectories(Self, FolderName, {SearchSubDirs}false);
  FixObjects();
  //WriteLn(FolderName,':',Self.Count);
  if Self.Count > 0 then
  begin
    MaxClass := Self.Count - 1;
    begin
      for ClassCnt := 0 to MaxClass do
      begin
        ClassFolder := Self[ClassCnt] + DirectorySeparator;
        if FImageSubFolder <> '' then
        begin
          ClassFolder := ClassFolder + FImageSubFolder + DirectorySeparator;
        end;
        if not Assigned(Self.List[ClassCnt]) then
        begin
          WriteLn(ClassFolder,' - error: not assigned list');
        end;
        FindAllFiles(Self.List[ClassCnt], ClassFolder, '*.png;*.jpg;*.jpeg;*.bmp;*.tif', {SearchSubDirs} false);
        Self.List[ClassCnt].FixObjects();
        ElementCnt := Self.List[ClassCnt].Count;
        SkipFirst := Round(ElementCnt * fSkipFirst);
        Loading := Round(ElementCnt * fLoadLen);
        SkipLast := ElementCnt - (Loading + SkipFirst);
        if SkipFirst > 0 then Self.List[ClassCnt].DeleteFirst(SkipFirst);
        if SkipLast > 0 then Self.List[ClassCnt].DeleteLast(SkipLast);
        //WriteLn(ClassFolder,':',Self.List[ClassCnt].Count);
      end;
    end;
  end;
end;

procedure TClassesAndElements.LoadImages(color_encoding: integer; NewSizeX: integer = 0; NewSizeY: integer = 0);
var
  NTL: TNeuralThreadList;
begin
  NTL := TNeuralThreadList.Create(Min(NeuralDefaultThreadCount(),Self.Count));
  if Self.Count > 0 then
  begin
    FNewSizeX := NewSizeX;
    FNewSizeY := NewSizeY;
    FColorEncoding := color_encoding;
    // start threads
    {$IFDEF Debug}
    Self.LoadImages_NTL(0,1);
    {$ELSE}
    NTL.StartProc({$IFDEF FPC}@{$ENDIF}Self.LoadImages_NTL);
    {$ENDIF}
  end;
  NTL.Free;
end;

procedure TClassesAndElements.LoadClass_FilenameFromFolder(FolderName: string);
begin
  // To Do
end;

function TClassesAndElements.GetRandomClassId(): integer;
begin
  if Self.Count > 0 then
  begin
    Result := Random(Self.Count);
  end
  else
  begin
    Result := -1;
  end;
end;

function TClassesAndElements.GetClassesCount(): integer;
begin
  result := Self.Count;
end;

procedure TClassesAndElements.GetRandomFileId(out ClassId: integer; out
  FileId:integer; StartPos: TNeuralFloat; Range: TNeuralFloat);
var
  StartPosId: integer;
  FileCnt: integer;
begin
  ClassId := GetRandomClassId();
  FileCnt := FileCountAtClassId(ClassId);
  StartPosId := Round(StartPos * FileCnt);
  FileId := Min(StartPosId + Random(Round(Range * FileCnt)), FileCnt-1);
end;

procedure TClassesAndElements.GetRandomFileName(out ClassId: integer; out
  FileName: string; StartPos: TNeuralFloat; Range: TNeuralFloat);
var
  FileId: integer;
begin
  GetRandomFileId(ClassId, FileId, StartPos, Range);
  FileName := GetFileName(ClassId, FileId);
end;

procedure TClassesAndElements.GetRandomImgVolumes(vInput,
  vOutput: TNNetVolume; StartPos: TNeuralFloat; Range: TNeuralFloat);
var
  ClassId: integer;
  FileName: string;
begin
  GetRandomFileName(ClassId, FileName, StartPos, Range);
  vOutput.Resize(GetClassesCount());
  vOutput.SetClassForSoftMax(ClassId);
  LoadImageFromFileIntoVolume(FileName, vInput);
end;

function TClassesAndElements.GetFileName(ClassId, ElementId: integer): string;
begin
  Result := Self.List[ClassId].Strings[ElementId];
end;

procedure TClassesAndElements.AddVolumesTo(Volumes: TNNetVolumeList; EmptySource:boolean = false);
var
  SourceVolume: TNNetVolume;
  ClassId, ImageId: integer;
  MaxClass, MaxImage: integer;
begin
  if Self.Count > 0 then
  begin
    MaxClass := Self.Count - 1;
    for ClassId := 0 to MaxClass do
    begin
      MaxImage := Self.List[ClassId].Count - 1;
      if MaxImage >= 0 then
      begin
        for ImageId := 0 to MaxImage do
        begin
          SourceVolume := Self.List[ClassId].List[ImageId];
          Volumes.AddCopy(SourceVolume);
          if EmptySource then SourceVolume.ReSize(1,1,1);
        end;
      end;
    end;
  end;
end;

procedure TClassesAndElements.AddFileNamesTo(FileNames: TFileNameList);
var
  SourceVolume: TNNetVolume;
  ClassId, FileId: integer;
  MaxClassId, MaxFileId: integer;
begin
  if Self.Count > 0 then
  begin
    MaxClassId := Self.Count - 1;
    for ClassId := 0 to MaxClassId do
    begin
      MaxFileId := Self.List[ClassId].Count - 1;
      if MaxFileId >= 0 then
      begin
        for FileId := 0 to MaxFileId do
        begin
          FileNames.AddInteger(GetFileName(ClassId, FileId), ClassId);
        end;
      end;
    end;
  end;
end;

procedure TClassesAndElements.MakeMonopolar(Divisor: TNeuralFloat);
var
  SourceVolume: TNNetVolume;
  ClassId, ImageId: integer;
  MaxClass, MaxImage: integer;
begin
  if Self.Count > 0 then
  begin
    MaxClass := Self.Count - 1;
    for ClassId := 0 to MaxClass do
    begin
      MaxImage := Self.List[ClassId].Count - 1;
      if MaxImage >= 0 then
      begin
        for ImageId := 0 to MaxImage do
        begin
          SourceVolume := Self.List[ClassId].List[ImageId];
          if Assigned(SourceVolume) then
          begin
            SourceVolume.Add(2);
            SourceVolume.Divi(Divisor);
          end;
        end;
      end;
    end;
  end;
end;

function TClassesAndElements.FileCountAtClassId(ClassId: integer): integer;
begin
  result := Self.List[ClassId].Count;
end;

{$IFDEF FPC}
function TFileNameList.ThreadSafeLoadImageFromFileIntoVolume(
  ImageFileName: string; V: TNNetVolume): boolean;
var
  M: TFPMemoryImage;
begin
  M := TFPMemoryImage.Create(1, 1);
  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSecLoad);{$ENDIF}
  Result := M.LoadFromFile( ImageFileName );
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSecLoad);{$ENDIF}
  if Result then LoadImageIntoVolume(M, V);
  M.Free;
end;

function LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume):boolean;
var
  M: TFPMemoryImage;
begin
  M := TFPMemoryImage.Create(1, 1);
  Result := M.LoadFromFile( ImageFileName );
  if Result then LoadImageIntoVolume(M, V);
  M.Free;
end;

function SaveImageFromVolumeIntoFile(V: TNNetVolume; ImageFileName: string
  ): boolean;
var
  M: TFPMemoryImage;
begin
  M := TFPMemoryImage.Create(V.SizeX, V.SizeY);
  LoadVolumeIntoImage(V, M);
  Result := M.SaveToFile(ImageFileName);
  M.Free;
end;

procedure LoadImageIntoVolume(M: TFPMemoryImage; Vol:TNNetVolume);
var
  CountX, CountY, MaxX, MaxY: integer;
  LocalColor: TFPColor;
  RawPos: integer;
begin
  MaxX := M.Width - 1;
  MaxY := M.Height - 1;
  Vol.ReSize(MaxX + 1, MaxY + 1, 3);

  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      LocalColor := M.Colors[CountX, CountY];
      RawPos := Vol.GetRawPos(CountX, CountY, 0);

      Vol.FData[RawPos]     := LocalColor.red shr 8;
      Vol.FData[RawPos + 1] := LocalColor.green shr 8;
      Vol.FData[RawPos + 2] := LocalColor.blue shr 8;
    end;
  end;
end;

procedure LoadVolumeIntoImage(Vol: TNNetVolume; M: TFPMemoryImage);
var
  CountX, CountY, MaxX, MaxY: integer;
  LocalColor: TFPColor;
  RawPos: integer;
begin
  MaxX := Vol.SizeX - 1;
  MaxY := Vol.SizeY - 1;
  M.SetSize(Vol.SizeX, Vol.SizeY);
  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      RawPos := Vol.GetRawPos(CountX, CountY, 0);
      LocalColor.red := NeuronForceMinMax(Round(Vol.FData[RawPos]),0,255) shl 8;
      LocalColor.green := NeuronForceMinMax(Round(Vol.FData[RawPos + 1]),0,255) shl 8;
      LocalColor.blue := NeuronForceMinMax(Round(Vol.FData[RawPos + 2]),0, 255) shl 8;
      M.Colors[CountX, CountY] := LocalColor;
    end;
  end;
end;
{$ELSE}
procedure LoadPictureIntoVolume(Picture: TPicture; Vol:TNNetVolume);
var
  CountX, CountY, MaxX, MaxY: integer;
  LocalColor: TColor;
  RawPos: integer;
begin
  MaxX := Picture.Width - 1;
  MaxY := Picture.Height - 1;
  Vol.ReSize(MaxX + 1, MaxY + 1, 3);

  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      LocalColor := Picture.Bitmap.Canvas.Pixels[CountX, CountY];
      RawPos := Vol.GetRawPos(CountX, CountY, 0);

      Vol.FData[RawPos]     := LocalColor and 255;
      Vol.FData[RawPos + 1] := (LocalColor shr 8) and 255;
      Vol.FData[RawPos + 2] := (LocalColor shr 16) and 255;
    end;
  end;
end;

function TFileNameList.ThreadSafeLoadImageFromFileIntoVolume(
  ImageFileName: string; V: TNNetVolume): boolean;
var
  LocalPicture: TPicture;
begin
  LocalPicture := TPicture.Create;
  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSecLoad);{$ENDIF}
  LocalPicture.LoadFromFile( ImageFileName );
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSecLoad);{$ENDIF}
  LoadPictureIntoVolume(LocalPicture, V);
  LocalPicture.Free;
end;

function LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume):boolean;
var
  LocalPicture: TPicture;
begin
  LocalPicture := TPicture.Create;
  LocalPicture.LoadFromFile( ImageFileName );
  LoadPictureIntoVolume(LocalPicture, V);
  LocalPicture.Free;
  Result := true;
end;

(*
function SaveImageFromVolumeIntoFile(V: TNNetVolume; ImageFileName: string
  ): boolean;
var
  LocalPicture: TPicture;
begin
  LocalPicture := TPicture.Create;
  LoadVolumeIntoImage(V, M);
  Result := M.SaveToFile(ImageFileName);
  LocalPicture.Free;
end;
*)
{$ENDIF}

procedure TClassesAndElements.LoadImages_NTL(index, threadnum: integer);
var
  SourceVolume: TNNetVolume;
  AuxVolume: TNNetVolume;
  ClassId, ImageId: integer;
  MaxClass, MaxImage: integer;
  {$IFDEF FPC}
  M: TFPMemoryImage;
  {$ELSE}
  LocalPicture: TPicture;
  {$ENDIF}
begin
  AuxVolume := TNNetVolume.Create();
  {$IFDEF FPC}
  M := TFPMemoryImage.Create(8,8);
  {$ELSE}
  LocalPicture := TPicture.Create;
  {$ENDIF}
  if Self.Count > 0 then
  begin
    MaxClass := Self.Count - 1;
    for ClassId := 0 to MaxClass do
    begin
      if ClassId mod threadnum = index then
      begin
        MaxImage := Self.List[ClassId].Count - 1;
        if MaxImage >= 0 then
        begin
          for ImageId := 0 to MaxImage do
          begin
            SourceVolume := Self.List[ClassId].List[ImageId];
            // Debug: WriteLn('Loading: ', Self.GetFileName(ClassId, ImageId));
            try
            {$IFDEF FPC}
              {$IFDEF HASTHREADS}EnterCriticalSection(FCritSecLoad);{$ENDIF}
              M.LoadFromFile( Self.GetFileName(ClassId, ImageId) );
              {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSecLoad);{$ENDIF}
              LoadImageIntoVolume(M, SourceVolume);
            {$ELSE}
              LocalPicture.LoadFromFile( Self.GetFileName(ClassId, ImageId) );
              LoadPictureIntoVolume(LocalPicture, SourceVolume);
            {$ENDIF}
            except
              WriteLn('Failed loading image: ',Self.GetFileName(ClassId, ImageId));
              SourceVolume.ReSize(FNewSizeX, FNewSizeY, 3);
              {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSecLoad);{$ENDIF}
            end;
            if (FNewSizeX > 0) and (FNewSizeY > 0) then
            begin
              if (SourceVolume.SizeX <> FNewSizeX) or (SourceVolume.SizeY <> FNewSizeY) then
              begin
                AuxVolume.Copy(SourceVolume);
                SourceVolume.CopyResizing(AuxVolume, FNewSizeX, FNewSizeY);
              end;
            end;
            SourceVolume.Tag := ClassId;
            SourceVolume.RgbImgToNeuronalInput(FColorEncoding);
          end;
        end;
      end;
    end;
  end;
  {$IFDEF FPC}
  M.Free;
  {$ELSE}
  LocalPicture.Free;
  {$ENDIF}
  AuxVolume.Free;
end;

procedure CreateFileNameListsFromImagesFromFolder(out TrainingFileNames,
  ValidationFileNames, TestFileNames: TFileNameList; FolderName,
  pImageSubFolder: string; TrainingProp,
  ValidationProp, TestProp: single);
var
  ClassesAndElements: TClassesAndElements;
begin
  TrainingFileNames := TFileNameList.Create();
  ValidationFileNames := TFileNameList.Create();
  TestFileNames := TFileNameList.Create();

  ClassesAndElements := TClassesAndElements.Create();

  if ValidationProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, TrainingProp, ValidationProp);
    ClassesAndElements.AddFileNamesTo(ValidationFileNames);
    ValidationFileNames.ClassCount := ClassesAndElements.GetClassesCount();
    ClassesAndElements.Clear;
  end;

  if TestProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, TrainingProp + ValidationProp, TestProp);
    ClassesAndElements.AddFileNamesTo(TestFileNames);
    TestFileNames.ClassCount := ClassesAndElements.GetClassesCount();
    ClassesAndElements.Clear;
  end;

  if TrainingProp > 0 then
  begin
    ClassesAndElements.LoadFoldersAsClassesProportional(FolderName, pImageSubFolder, 0, TrainingProp);
    ClassesAndElements.AddFileNamesTo(TrainingFileNames);
    TrainingFileNames.ClassCount := ClassesAndElements.GetClassesCount();
    ClassesAndElements.Clear;
  end;

  ClassesAndElements.Free;
end;

procedure TranslateCifar10VolumesToMachineAnimal(VolumeList: TNNetVolumeList);
var
  Volume: TNNetVolume;
begin
  for Volume in VolumeList do
  begin
    Volume.Tag := csMachineAnimalCifar10Pos[Volume.Tag];
  end;
end;

procedure CreateCifar10Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  ValidationSampleSize: integer = 2000);
var
  I, LastElement: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();

  if ValidationSampleSize > 10000 then ValidationSampleSize := 10000;
  if ValidationSampleSize < 0 then ValidationSampleSize := 0;

  // creates required volumes to store images
  for I := 0 to 39999 do
  begin
    ImgTrainingVolumes.Add(TNNetVolume.Create());
  end;

  for I := 0 to 9999 do
  begin
    ImgValidationVolumes.Add(TNNetVolume.Create());
    ImgTestVolumes.Add(TNNetVolume.Create());
  end;

  loadCifar10Dataset(ImgTrainingVolumes, 1, 0, color_encoding);
  loadCifar10Dataset(ImgTrainingVolumes, 2, 10000, color_encoding);
  loadCifar10Dataset(ImgTrainingVolumes, 3, 20000, color_encoding);
  loadCifar10Dataset(ImgTrainingVolumes, 4, 30000, color_encoding);
  loadCifar10Dataset(ImgValidationVolumes, 5, 0, color_encoding);
  loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0, color_encoding);

  // Should move validation volumes to training volumes?
  if ValidationSampleSize < 10000 then
  begin
    ImgValidationVolumes.FreeObjects := False;
    LastElement := ImgValidationVolumes.Count - 1;
    for I := LastElement downto (LastElement-(10000-ValidationSampleSize)+1) do
    begin
      ImgTrainingVolumes.Add(ImgValidationVolumes[I]);
      ImgValidationVolumes.Delete(I);
    end;
    ImgValidationVolumes.FreeObjects := True;
  end;
end;

procedure CreateCifar100Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  Verbose:boolean = true; ValidationSampleSize: integer = 2000);
var
  I, LastElement: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  loadCifar100Dataset(ImgTrainingVolumes, 'train.bin', color_encoding, Verbose);
  loadCifar100Dataset(ImgTestVolumes, 'test.bin', color_encoding, Verbose);
  LastElement := ImgTrainingVolumes.Count - 1;
  if ValidationSampleSize > 0 then
  begin
    ImgTrainingVolumes.FreeObjects := false;
    if ValidationSampleSize > 25000 then ValidationSampleSize := 25000;
    for I := LastElement downto (LastElement-ValidationSampleSize+1) do
    begin
      ImgValidationVolumes.Add(ImgTrainingVolumes[I]);
      ImgTrainingVolumes.Delete(I);
    end;
    ImgTrainingVolumes.FreeObjects := true;
  end;
end;

procedure CreateMNISTVolumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; TrainFileName, TestFileName: string;
  Verbose: boolean = true; IsFashion:boolean = false);
var
  I, HalfSize, LastElement: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  loadMNISTDataset(ImgTrainingVolumes, TrainFileName, Verbose, IsFashion);
  loadMNISTDataset(ImgValidationVolumes, TestFileName, Verbose, IsFashion);
  ImgValidationVolumes.FreeObjects := false;
  HalfSize := ImgValidationVolumes.Count div 2;
  LastElement := ImgValidationVolumes.Count - 1;
  for I := LastElement downto HalfSize do
  begin
    ImgTestVolumes.Add(ImgValidationVolumes[I]);
    ImgValidationVolumes.Delete(I);
  end;
  ImgValidationVolumes.FreeObjects := true;
end;

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; idx:integer; base_pos:integer = 0; color_encoding: byte = csEncodeRGB);
var
  fileName: string;
begin
  fileName := 'data_batch_'+IntToStr(idx)+'.bin';
  loadCifar10Dataset(ImgVolumes, fileName, base_pos, color_encoding);
end;

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; fileName:string; base_pos:integer = 0; color_encoding: byte = csEncodeRGB);
var
  I, ImgPos: integer;
  Img: TTinyImage;
  cifarFile: TTInyImageFile;
  AuxVolume: TNNetVolume;
  pMin, pMax: TNeuralFloat;
  globalMin0, globalMax0: TNeuralFloat;
  globalMin1, globalMax1: TNeuralFloat;
  globalMin2, globalMax2: TNeuralFloat;

begin
  Write('Loading 10K images from file "'+fileName+'" ...');
  AssignFile(cifarFile, fileName);
  Reset(cifarFile);
  AuxVolume := TNNetVolume.Create();

  globalMin0 := 0;
  globalMax0 := 0;
  globalMin1 := 0;
  globalMax1 := 0;
  globalMin2 := 0;
  globalMax2 := 0;

  // binary CIFAR 10 file contains 10K images
  for I := 0 to 9999 do
  begin
    Read(cifarFile, Img);
    ImgPos := I + base_pos;
    LoadTinyImageIntoNNetVolume(Img, ImgVolumes[ImgPos]);

    if (color_encoding = csEncodeGray) then
    begin
      AuxVolume.Copy(ImgVolumes[ImgPos]);
      ImgVolumes[ImgPos].GetGrayFromRgb(AuxVolume);
    end;

    ImgVolumes[ImgPos].RgbImgToNeuronalInput(color_encoding);

    ImgVolumes[ImgPos].GetMinMaxAtDepth(0, pMin, pMax); //WriteLn  (I:8,' - #0 Min:',pMin, ' Max:',pMax);

    globalMin0 := Math.Min(pMin, globalMin0);
    globalMax0 := Math.Max(pMax, globalMax0);

    if (ImgVolumes[ImgPos].Depth >= 2) then
    begin
      ImgVolumes[ImgPos].GetMinMaxAtDepth(1, pMin, pMax); //Write  (' #1 Min:',pMin, ' Max:',pMax);

      globalMin1 := Math.Min(pMin, globalMin1);
      globalMax1 := Math.Max(pMax, globalMax1);
    end;

    if (ImgVolumes[ImgPos].Depth >= 3) then
    begin
      ImgVolumes[ImgPos].GetMinMaxAtDepth(2, pMin, pMax); //WriteLn(' #2 Min:',pMin, ' Max:',pMax);

      globalMin2 := Math.Min(pMin, globalMin2);
      globalMax2 := Math.Max(pMax, globalMax2);
    end;
  end;

  Write(' GLOBAL MIN MAX ', globalMin0:8:4,globalMax0:8:4,globalMin1:8:4,globalMax1:8:4,globalMin2:8:4,globalMax2:8:4);

  AuxVolume.Free;
  CloseFile(cifarFile);
  WriteLn(' Done.');
end;

procedure loadCifar100Dataset(ImgVolumes: TNNetVolumeList; fileName: string;
  color_encoding: byte = csEncodeRGB; Verbose:boolean = true);
var
  Img: TCifar100Image;
  cifarFile: TCifar100File;
  AuxVolume: TNNetVolume;
begin
  if Verbose then Write('Loading images from CIFAR-100 file "'+fileName+'" ...');
  AssignFile(cifarFile, fileName);
  Reset(cifarFile);

  // binary CIFAR 10 file contains 10K images
  while not(Eof(cifarFile)) do
  begin
    AuxVolume := TNNetVolume.Create();
    Read(cifarFile, Img);
    LoadTinyImageIntoNNetVolume(Img, AuxVolume);
    AuxVolume.RgbImgToNeuronalInput(color_encoding);
    ImgVolumes.Add(AuxVolume);

    //TODO: add treatment for CIFAR100
    (*if (color_encoding = csEncodeGray) then
    begin
      AuxVolume.Copy(ImgVolumes[ImgPos]);
      ImgVolumes[ImgPos].GetGrayFromRgb(AuxVolume);
    end;*)
  end;

  CloseFile(cifarFile);
  if Verbose then WriteLn(' ',ImgVolumes.Count, ' images loaded.');
end;

procedure loadMNISTDataset(ImgVolumes: TNNetVolumeList; fileName: string;
  Verbose: boolean = true; IsFashion:boolean = false;
  MaxLabel: integer = 10);
var
  fileNameLabels, fileNameImg: string;
  fileLabels, fileImg: THandle;
  LabelItems, LabelMagic: integer;
  ImgMagic, ImgItems, ImgRows, ImgCols: Integer;
  MNistImg: TMNistImage;
  ImgCnt: integer;
  LabelByte: byte;
  Vol: TNNetVolume;
  Separator: char;
begin
  if IsFashion
    then Separator := '-'
    else Separator := '.';

  fileNameLabels := fileName + '-labels'+Separator+'idx1-ubyte';
  fileNameImg := fileName + '-images'+Separator+'idx3-ubyte';

  if not FileExists(fileNameLabels) then
  begin
    if Verbose then WriteLn('Labels file not found:', fileNameLabels);
    exit;
  end;

  if not FileExists(fileNameImg) then
  begin
    if Verbose then WriteLn('Image file not found:', fileNameImg);
    exit;
  end;

  fileLabels := FileOpen(fileNameLabels, fmOpenRead);
  fileImg := FileOpen(fileNameImg, fmOpenRead);

  FileRead(fileLabels, LabelMagic, 4);
  FileRead(fileLabels, LabelItems, 4);

  FileRead(fileImg, ImgMagic, 4);
  FileRead(fileImg, ImgItems, 4);
  FileRead(fileImg, ImgRows, 4);
  FileRead(fileImg, ImgCols, 4);

  LabelItems := SwapEndian(LabelItems);
  LabelMagic := SwapEndian(LabelMagic);
  ImgMagic := SwapEndian(ImgMagic);
  ImgItems := SwapEndian(ImgItems);
  ImgRows := SwapEndian(ImgRows);
  ImgCols := SwapEndian(ImgCols);

  if Verbose then
    WriteLn
    (
      'File:', fileName,
      ' Labels:', LabelItems, ' Images:', ImgItems,
      ' Rows:', ImgRows, ' Cols:', ImgCols
    );

  for ImgCnt := 1 to ImgItems do
  begin
    FileRead(fileImg, MNistImg, SizeOf(TMNistImage));
    FileRead(fileLabels, LabelByte, 1);
    Vol := TNNetVolume.Create();
    LoadTinyImageIntoNNetVolume(MNistImg, Vol);
    Vol.Divi(64);
    Vol.Add(-2);
    Vol.Tag := LabelByte;
    if LabelByte >= MaxLabel then
    begin
      WriteLn('Error loading Label:', LabelByte,' at index ', ImgCnt);
      Vol.Free;
    end
    else
    begin
      ImgVolumes.Add(Vol);
    end;
  end;

  FileClose(fileLabels);
  FileClose(fileImg);
end;

// This function returns TRUE if data_batch_1.bin and error message otherwise
function CheckCIFARFile():boolean;
begin
  Result := true;
  if not (FileExists('data_batch_1.bin')) then
  begin
    WriteLn('File Not Fount: data_batch_1.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end;
end;

function CheckCIFAR100File(): boolean;
begin
  Result := true;
  if not (FileExists('train.bin')) then
  begin
    WriteLn('File Not Fount: train.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end
  else if not (FileExists('test.bin')) then
  begin
    WriteLn('File Not Fount: test.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end;
end;

function CheckMNISTFile(fileName: string; IsFasion:boolean = false): boolean;
var
  Separator: char;
  Link: string;
  FullFileName: string;
begin
  Result := true;
  if IsFasion then
  begin
    Separator := '-';
    Link := 'https://github.com/zalandoresearch/fashion-mnist';
  end
  else
  begin
    Separator := '.';
    Link := 'http://yann.lecun.com/exdb/mnist/';
  end;
  FullFileName := fileName+'-labels'+Separator+'idx1-ubyte';
  if not (FileExists(FullFileName)) then
  begin
    WriteLn('File Not Fount:', FullFileName);
    WriteLn('Please download from ', Link);
    //TODO: automatically download file
    Result := false;
    exit;
  end;
  FullFileName := fileName+'-images'+Separator+'idx3-ubyte';
  if not (FileExists(FullFileName)) then
  begin
    WriteLn('File Not Fount:', FullFileName);
    WriteLn('Please download from ', Link);
    //TODO: automatically download file
    Result := false;
  end;
end;

function LoadImageFromFileIntoVolume(ImageFileName:string; V:TNNetVolume;
  SizeX, SizeY: integer;
  EncodeNeuronalInput: integer = -1
  ): boolean;
var
  VAux: TNNetVolume;
begin
  if LoadImageFromFileIntoVolume(ImageFileName, V) then
  begin
    if (V.SizeX<>SizeX) or (V.SizeY<>SizeY) then
    begin
      VAux := TNNetVolume.Create;
      VAux.Copy(V);
      V.CopyResizing(VAux, SizeX, SizeY);
      VAux.Free;
    end;
    if (EncodeNeuronalInput >= 0) then
    begin
      V.RgbImgToNeuronalInput( (EncodeNeuronalInput) and 255 );
    end;
    Result := true;
  end
  else
  begin
    Result := false;
  end;
end;

procedure ConfusionWriteCSVHeader(var CSVConfusion: TextFile; Labels: array of string);
var
  I: integer;
begin
  for I := Low(Labels) to High(Labels) do
  begin
    if I > 0 then Write(CSVConfusion, ',');
    Write(CSVConfusion, Labels[I]);
  end;
  WriteLn(CSVConfusion);
end;

procedure ConfusionWriteCSV(var CSVConfusion: TextFile; Vol: TNNetVolume; Digits: integer);
var
  I, J: integer;
begin
  for I := 0 to Vol.SizeY - 1 do
  begin
    for J := 0 to Vol.Depth - 1 do
    begin
      if J > 0 then Write(CSVConfusion, ',');
      Write(CSVConfusion, Round(Vol[0, I, J]):Digits);
    end;
    WriteLn(CSVConfusion);
  end;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TTinyImage; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,3);
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[J, I, 0] := TI.R[I, J];
      Vol[J, I, 1] := TI.G[I, J];
      Vol[J, I, 2] := TI.B[I, J];
    end;
  end;
  Vol.Tag := TI.bLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TCifar100Image; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,3);
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[J, I, 0] := TI.R[I, J];
      Vol[J, I, 1] := TI.G[I, J];
      Vol[J, I, 2] := TI.B[I, J];
    end;
  end;
  Vol.Tags[0] := TI.bFineLabel;
  Vol.Tags[1] := TI.bCoarseLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TMNistImage; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(28, 28, 1);
  for I := 0 to 27 do
  begin
    for J := 0 to 27 do
    begin
      Vol[J, I, 0] := TI[I, J];
    end;
  end;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TTinyImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TI.R[I, J] := Vol.AsByte[J, I, 0];
      TI.G[I, J] := Vol.AsByte[J, I, 1];
      TI.B[I, J] := Vol.AsByte[J, I, 2];
    end;
  end;
  TI.bLabel := Vol.Tag;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TCifar100Image);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TI.R[I, J] := Vol.AsByte[J, I, 0];
      TI.G[I, J] := Vol.AsByte[J, I, 1];
      TI.B[I, J] := Vol.AsByte[J, I, 2];
    end;
  end;
  TI.bCoarseLabel := Vol.Tags[0];
  TI.bFineLabel := Vol.Tags[1];
end;

procedure LoadTinySingleChannelIntoNNetVolume(var SC: TTinySingleChannelImage;
  Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,1);
  Vol.Tag := SC.bLabel;
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[I, J, 0] := SC.Grey[I,J];
    end;
  end;
end;

procedure TinyImageCreateGrey(var TI: TTinyImage; var TIGrey: TTinySingleChannelImage);
var
  I, J: integer;
begin
  TIGrey.bLabel := TI.bLabel;
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TIGrey.Grey[I, J] := (TI.R[I, J] + TI.G[I, J] + TI.B[I, J]) div 3;
    end;
  end;
end;

procedure TinyImageVE(var TI, TIVE: TTinySingleChannelImage);
var
  I, J: integer;
  aux: integer;
begin
  TIVE.bLabel := TI.bLabel;
  for I := 1 to 31 do
  begin
    TIVE.Grey[0, I] := 128;
    for J := 0 to 31 do
    begin
      aux := (TI.Grey[I, J] - TI.Grey[I - 1, J]) * 3 + 128;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIVE.Grey[I, J] := aux;
    end;
  end;
end;

procedure TinyImageHE(var TI, TIHE: TTinySingleChannelImage);
var
  I, J: integer;
  aux: integer;
begin
  TIHE.bLabel := TI.bLabel;
  for I := 0 to 31 do
  begin
    TIHE.Grey[I, 0] := 128;
    for J := 1 to 31 do
    begin
      aux := (TI.Grey[I, J] - TI.Grey[I, J - 1]) * 3 + 128;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIHE.Grey[I, J] := aux;
    end;
  end;
end;

procedure TinyImageRemoveZeroGradient(var TI: TTinySingleChannelImage; distance: byte);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      if abs(TI.Grey[I, J] - 128) < distance then
        TI.Grey[I, J] := 0;
    end;
  end;
end;

procedure TinyImageHVE(var TI, TIHE: TTinySingleChannelImage);
var
  I, J: integer;
  aux, aux1, aux2: integer;
begin
  TIHE.bLabel := TI.bLabel;
  for I := 1 to 31 do
  begin
    TIHE.Grey[I, 0] := 128;
    TIHE.Grey[0, I] := 128;
    for J := 1 to 31 do
    begin
      aux1 := (TI.Grey[I, J] - TI.Grey[I, J - 1]) + 128;
      aux2 := (TI.Grey[I, J] - TI.Grey[I - 1, J]) + 128;

      if (abs(aux1 - 128) > abs(aux2 - 128)) then
      begin
        aux := aux1;
      end
      else
      begin
        aux := aux2;
      end;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIHE.Grey[I, J] := aux;
    end;
  end;
end;

function TinyImageTo1D(var TI: TTinySingleChannelImage): TTinySingleChannelImage1D;
var
  TIPtr: TTinySingleChannelImage1DPtr;
begin
  TIPtr := addr(TI);
  Result := TIPtr^;
end;

// this function tests a neural network on the passed ImgVolumes
procedure TestBatch
(
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
);
var
  I, ImgIdx: integer;
  hit, miss: integer;
  pOutput, vOutput: TNNetVolume;
  bIsSoftmax: boolean;
  CurrentLoss : TNeuralFloat;
  OutputValue: TNeuralFloat;
  MaxID: integer;
begin
  pOutput := TNNetVolume.Create;
  vOutput := TNNetVolume.Create;

  hit  := 0;
  miss := 0;
  ErrorSum := 0;
  Loss := 0;
  Rate := 0;
  bIsSoftmax := false;

  if NN.Layers[NN.GetLastLayerIdx()] is TNNetSoftMax then
  begin
    bIsSoftmax := true;
  end;

  if SampleSize = 0 then
  begin
    MaxID := ImgVolumes.Count;
  end
  else
  begin
    MaxID := SampleSize;
  end;

  for I := 1 to MaxID do
  begin
    if SampleSize = 0 then
    begin
      ImgIdx := I - 1;
    end
    else
    begin
      ImgIdx := Random(ImgVolumes.Count);
    end;

    NN.Compute(ImgVolumes[ImgIdx]);
    NN.GetOutput(pOutput);

    ImgVolumes[ImgIdx].FlipX();
    NN.AddOutput(pOutput);

    if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then
    begin
      Inc(Hit);
    end
    else
    begin
      Inc(Miss);
    end;

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( ImgVolumes[ImgIdx].Tag );
    end
    else
    begin
      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag );
    end;

    ErrorSum := ErrorSum + vOutput.SumDiff(pOutput);

    if (bIsSoftmax) then
    begin
      OutputValue := pOutput.FData[ ImgVolumes[ImgIdx].Tag ];
      if (OutputValue > 0) then
      begin
        CurrentLoss := -Ln(OutputValue);
      end
      else
      begin
        WriteLn('Error: invalid output value',OutputValue);
        CurrentLoss := 1;
      end;

      Loss := Loss + CurrentLoss;
    end;
  end;

  if (Hit > 0) then
  begin
    Rate := Hit / (Hit + Miss);
    Loss := Loss / (Hit + Miss);
  end;

  vOutput.Free;
  pOutput.Free;
end;

function RemoveRandomWord(const Str: string): string;
var
  WordList: TNNetStringList;
  RandomWordIndex: integer;
begin
  Result := Str;
  // Split the string into words based on spaces.
  WordList := CreateTokenizedStringList(Result,' ');
  // Check if there are any words to remove.
  if WordList.Count > 1 then
  begin
    // Select a random word to remove.
    RandomWordIndex := Random(WordList.Count);
    WordList.Delete(RandomWordIndex);
    // Reconstruct the string from the remaining words.
    Result := WordList.DelimitedText;
  end;
  // Free the TStringList to prevent memory leaks.
  WordList.Free;
end;

procedure FilterCSVWithNumbersUpToMax(inputFile,outputFile: string; MaxInteger: integer; MaxRows: integer = 0);
var
  LargeFileIn, LargeFileOut: TextFile;
  StrLine: string;
  MaxValue, RowCnt, WordCnt: integer;
  Separator: TNNetStringList;
begin
  Separator := CreateTokenizedStringList(',');
  RowCnt := 0;
  //WriteLn('Counting rows from: ', filename);
  AssignFile(LargeFileIn, inputFile);
  AssignFile(LargeFileOut, outputFile);
  Reset(LargeFileIn);
  Rewrite(LargeFileOut);
  while (not Eof(LargeFileIn)) and ( (MaxRows=0) or (RowCnt<MaxRows) ) do
  begin
    ReadLn(LargeFileIn, StrLine);
    Separator.DelimitedText := StrLine;
    if Separator.Count > 0 then
    begin
      MaxValue := 0;
      for WordCnt := 0 to Separator.Count - 1 do
      begin
        MaxValue := Max(MaxValue, StrToInt(Separator[WordCnt]));
        if MaxValue > MaxInteger then break;
      end;
      if MaxValue <= MaxInteger then
      begin
        RowCnt := RowCnt + 1;
        WriteLn(LargeFileOut, StrLine);
      end;
    end;
  end;
  CloseFile(LargeFileIn);
  CloseFile(LargeFileOut);
end;

procedure LoadIntegersInCSV(filename: string; var aTokens: TNNetAAInteger;
  MaxRows: integer = 0);
var
  LargeFile: TextFile;
  StrLine: string;
  RowCnt, WordCnt: integer;
  Separator: TNNetStringList;
begin
  Separator := CreateTokenizedStringList(',');
  RowCnt := 0;
  //WriteLn('Counting rows from: ', filename);
  AssignFile(LargeFile, filename);
  Reset(LargeFile);
  while (not Eof(LargeFile)) and ( (MaxRows=0) or (RowCnt<MaxRows) ) do
  begin
    ReadLn(LargeFile, StrLine);
    RowCnt := RowCnt + 1;
  end;
  CloseFile(LargeFile);
  //WriteLn('Loading: ', filename);
  SetLength(aTokens, RowCnt);
  //WriteLn('Loading ', RowCnt,' rows.');
  Reset(LargeFile);
  RowCnt := 0;
  while (not Eof(LargeFile)) and ( (MaxRows=0) or (RowCnt<MaxRows) ) do
  begin
    ReadLn(LargeFile, StrLine);
    Separator.DelimitedText := StrLine;
    SetLength(aTokens[RowCnt], Separator.Count);
    if Separator.Count > 0 then
    begin
      for WordCnt := 0 to Separator.Count - 1 do
      begin
        aTokens[RowCnt][WordCnt] := StrToInt(Separator[WordCnt]);
      end;
    end;
    RowCnt := RowCnt + 1;
  end;
  CloseFile(LargeFile);
end;

function RemoveRandomChars(const Str: string; Count: integer): string;
var
  i: integer;
  StrLen: integer;
begin
  Result := Str;
  // Calculate the length of the string before removing characters.
  StrLen := Length(Result);
  if (Count > 0) and (StrLen>1) then
  begin
    // Loop for the number of characters to be removed.
    for i := 1 to Count do
    begin
      // Check if the string is not empty.
      if StrLen > 1 then
      begin
        // Randomly select a character position and remove one character from that position.
        // The '+ 1' is necessary because Pascal strings are 1-indexed, not 0-indexed.
        Delete(Result, Random(StrLen) + 1, 1);
        Dec(StrLen);
      end;
    end;
  end;
end;


function RandomSubstring(const InputString: string): string;
var
  SpacePositions: TIntegerList;
  I, RandomSpacePos: Integer;
  InputStringLen: integer;
begin
  InputStringLen := Length(InputString);
  if InputStringLen > 0 then
  begin
    // Create a new integer list instance
    SpacePositions := TIntegerList.Create;
    // Find the positions of all spaces in the string
    for I := 1 to InputStringLen do
    begin
      if InputString[I] = ' ' then
      begin
        SpacePositions.Add(I);
      end;
    end;

    // Append -1 to handle the case with no spaces
    SpacePositions.Add(0);

    // Randomly select one of the space positions
    RandomSpacePos := SpacePositions[Random(SpacePositions.Count)];

    // Return the substring starting from the position after the random space
    Result := Copy(InputString, RandomSpacePos + 1, InputStringLen - RandomSpacePos);
    SpacePositions.Free;
  end
  else Result := '';
end;

end.
