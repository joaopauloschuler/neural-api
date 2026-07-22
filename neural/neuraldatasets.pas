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
  {$IFNDEF FPC}System.Classes, Windows, Vcl.Graphics, System.JSON, System.Generics.Collections, {$ENDIF}
  neuralvolume, neuralnetwork, pascoremath32
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

// ---------------------------------------------------------------------------
// Standard vision-model image preprocessing (the timm/torchvision/CLIP
// inference transform). Plain LoadImageFromFileIntoVolume only does a stretch
// resize to (SizeX,SizeY); the helpers below do the canonical pipeline every
// vision importer expects:
//   (1) aspect-ratio-preserving resize of the SHORTER side to ResizeShorterSide
//   (2) center-crop to (CropSize x CropSize)
//   (3) per-channel normalization (x/255 - mean)/std.
// Mean/Std are 3-element RGB arrays; the published defaults are below.
// ---------------------------------------------------------------------------
const
  // torchvision/timm ImageNet normalization (the classifier default).
  csImageNetMean: array[0..2] of TNeuralFloat = (0.485, 0.456, 0.406);
  csImageNetStd:  array[0..2] of TNeuralFloat = (0.229, 0.224, 0.225);
  // OpenAI CLIP normalization.
  csClipMean: array[0..2] of TNeuralFloat =
    (0.48145466, 0.4578275, 0.40821073);
  csClipStd:  array[0..2] of TNeuralFloat =
    (0.26862954, 0.26130258, 0.27577711);

type
  // Scalars read straight from a HuggingFace preprocessor_config.json so the
  // transform drops into the importers (image classifiers, CLIP, ViT, ...).
  TNNetImagePreprocess = record
    ResizeShorterSide: integer;     // size.shortest_edge (resize target)
    CropSize: integer;              // crop_size (square)
    Mean: array[0..2] of TNeuralFloat;  // image_mean (RGB)
    Std: array[0..2] of TNeuralFloat;   // image_std (RGB)
  end;

// Applies the standard inference transform to Src (an (W,H,3) RGB volume with
// 0..255 byte-valued channels, the layout LoadImageFromFileIntoVolume yields)
// and writes the (CropSize,CropSize,3) normalized result into Dst:
//   shorter-side resize to ResizeShorterSide (aspect preserved, bilinear) ->
//   center-crop to CropSize -> (x/255 - Mean[c])/Std[c] per channel.
// Mean/Std are RGB triples (e.g. csImageNetMean / csImageNetStd). Src and Dst
// may NOT be the same volume.
procedure PreprocessImageForVisionModel(Src, Dst: TNNetVolume;
  ResizeShorterSide, CropSize: integer;
  const Mean, Std: array of TNeuralFloat);

// Reads the standard inference transform straight into a TNNetImagePreprocess
// from a HuggingFace preprocessor_config.json. Handles size as an int, as
// {"shortest_edge":N} or as {"height":..,"width":..}, and crop_size as an int
// or {"height":..,"width":..}. Defaults follow CLIPImageProcessor (shortest
// edge / crop 224, the OpenAI image_mean/image_std) when a field is absent.
function ReadImagePreprocessConfig(
  const FileName: string): TNNetImagePreprocess;

// Convenience: load an image file straight to a vision-model-ready normalized
// volume (LoadImageFromFileIntoVolume + PreprocessImageForVisionModel).
function LoadImageForVisionModel(const ImageFileName: string; V: TNNetVolume;
  ResizeShorterSide, CropSize: integer;
  const Mean, Std: array of TNeuralFloat): boolean;

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

// ----------------------------------------------------------------------------
//  RandAugment / TrivialAugment automatic augmentation policy
// ----------------------------------------------------------------------------
//  Single-image geometric/photometric augmentation, ported from the
//  torchvision transforms-v2 staple. The op bank and both selection policies
//  operate IN PLACE on a TNNetVolume holding an RGB(/gray) image in this
//  library's neuronal input domain ([-2..2], i.e. (pixel-128)/64). All ops
//  keep the result clamped to that domain.
//
//  Magnitude convention follows torchvision RandAugment: an integer M in
//  0..NeuralAugMaxMagnitude (default 10), 0 being (close to) identity. Each
//  op maps M onto its torchvision _AUGMENTATION_SPACE range.
// ----------------------------------------------------------------------------

const
  // Maximum magnitude value (torchvision uses num_magnitude_bins=31 -> M in
  // 0..30 by default; RandAugment's default magnitude is 9 on that scale.
  // We keep the same 0..30 scale).
  NeuralAugMaxMagnitude: integer = 30;

type
  // Identifiers for the single-image augmentation op bank. csaIdentity is a
  // genuine no-op; the geometric/photometric ops map magnitude M as described
  // above.
  TNeuralAugOp = (
    csaIdentity,
    csaAutoContrast,
    csaEqualize,
    csaRotate,
    csaShearX,
    csaShearY,
    csaTranslateX,
    csaTranslateY,
    csaPosterize,
    csaSolarize,
    csaColor,
    csaContrast,
    csaBrightness,
    csaSharpness
  );

// Applies a single augmentation op IN PLACE on V (neuronal [-2..2] image).
// Magnitude is the integer M on the 0..NeuralAugMaxMagnitude scale. For
// signed geometric ops (rotate/shear/translate) the sign is chosen randomly
// (torchvision behaviour) unless pSignedNeg is supplied. Photometric ops are
// clamped to the valid image domain.
procedure NeuralAugApplyOp(V: TNNetVolume; Op: TNeuralAugOp; Magnitude: integer);

// RandAugment (Cubuk et al. 2020): apply N ops drawn uniformly from the bank,
// each at the SAME fixed magnitude M. Operates IN PLACE on V. Deterministic
// for a fixed RandSeed.
procedure NeuralRandAugment(V: TNNetVolume; N: integer = 2; Magnitude: integer = 9);

// TrivialAugment (Muller & Hutter 2021): apply exactly ONE op drawn uniformly
// from the bank, with its magnitude drawn uniformly from 0..NeuralAugMaxMagnitude.
// Parameter-free. Operates IN PLACE on V. Deterministic for a fixed RandSeed.
procedure NeuralTrivialAugment(V: TNNetVolume);

// RandomErasing / Cutout (Zhong et al. 2020 / DeVries & Taylor 2017): with
// probability pProb, erase a random rectangle whose area is a fraction in
// [pAreaLow,pAreaHigh] of the image (aspect in [pAspectLow,1/pAspectLow]) by
// filling it with pFill (in the neuronal domain; default 0 = neutral gray).
// Operates IN PLACE. Deterministic for a fixed RandSeed.
procedure NeuralRandomErasing(V: TNNetVolume;
  pProb: TNeuralFloat = 0.5;
  pAreaLow: TNeuralFloat = 0.02; pAreaHigh: TNeuralFloat = 0.33;
  pAspectLow: TNeuralFloat = 0.3; pFill: TNeuralFloat = 0.0);

type
  // Selection policy for the optional TNeuralImageFit augmentation hook.
  TNeuralAugPolicy = (napNone, napRandAugment, napTrivialAugment);

  // TNeuralAugmentationPolicy is a tiny, stateless-per-call helper that bundles
  // a chosen policy (RandAugment / TrivialAugment) plus an optional
  // RandomErasing pass into a single procedure-of-object compatible with
  // TNNetDataAugmentationFn (see neuralfit.pas). Assign its Augment method to
  // TNeuralImageFit.DataAugmentationFn to opt CIFAR examples into the policy
  // WITHOUT disturbing the default flip+crop pipeline.
  // Coded by Claude (AI).
  TNeuralAugmentationPolicy = class(TObject)
  private
    FPolicy: TNeuralAugPolicy;
    FNumOps: integer;
    FMagnitude: integer;
    FErasingProb: TNeuralFloat;
  public
    constructor Create(pPolicy: TNeuralAugPolicy = napTrivialAugment;
      pNumOps: integer = 2; pMagnitude: integer = 9;
      pErasingProb: TNeuralFloat = 0.25);
    // Signature matches TNNetDataAugmentationFn (pInput, ThreadId).
    procedure Augment(pInput: TNNetVolume; ThreadId: integer);
    property Policy: TNeuralAugPolicy read FPolicy write FPolicy;
    property NumOps: integer read FNumOps write FNumOps;
    property Magnitude: integer read FMagnitude write FMagnitude;
    property ErasingProb: TNeuralFloat read FErasingProb write FErasingProb;
  end;

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
  // By default attention may cross document boundaries inside a packed window
  // (the standard GPT-2/GPT-3 packing behaviour, which works fine in practice).
  // To PREVENT cross-document attention, call GetSegmentIds / GetSegmentVolume
  // to obtain the per-token document ids and feed them as the segment source of
  // a TNNetScaledDotProductAttention layer (its optional per-sample
  // block-diagonal mask: query i attends key j only when seg[i] = seg[j]).
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
      // Per-token DOCUMENT/SEGMENT ids for window WindowIdx (length ContextLen),
      // for feeding TNNetScaledDotProductAttention's per-sample segment mask so
      // attention does NOT cross document boundaries inside the packed window.
      // A new document begins immediately after each end-of-document separator,
      // so the id is incremented every time the PREVIOUS token was a separator;
      // the separator itself shares its document's id (it is that document's
      // last token). Every trailing pad position is given one shared id that is
      // distinct from all real-document ids (their outputs are loss-masked, so
      // attending among themselves is harmless, while they never see a real
      // document). Returns a TNeuralIntegerArray of length ContextLen.
      function GetSegmentIds(WindowIdx: integer): TNeuralIntegerArray;
      // Fills pSegment (SizeX=ContextLen, SizeY=1, Depth=1) with the per-token
      // segment ids from GetSegmentIds, ready to be the segment source of a
      // TNNetScaledDotProductAttention layer.
      procedure GetSegmentVolume(WindowIdx: integer; pSegment: TNNetVolume);
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
      function IsSpecial(TokenId: integer): boolean;
      function SampleSpanLength(): integer;

    // <TEMP> Temporarily place unused functions in the protected section to avoid hints.
    protected
      function NextRandomInt(N: integer): integer;
    // </TEMP>

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

  { TNNetLengthGroupedBatcher }
  // Length-grouped batching with dynamic (per-batch) padding for variable-length
  // sequence training. A port of HuggingFace transformers LengthGroupedSampler
  // plus DataCollatorWithPadding: it is the TRAINING data-side throughput
  // optimization (distinct from per-sample attention masks or left-padded
  // generation). Instead of padding every sample to the GLOBAL maximum length,
  // it batches samples of SIMILAR length together and pads each emitted batch
  // only to THAT batch's own maximum length, which sharply reduces the wasted
  // pad-token compute on a length-skewed corpus.
  //
  // Ordering is the transformers "megabatch" shuffle, which keeps the data
  // stochastic across epochs while still grouping by length:
  //   1. shuffle all sample indices,
  //   2. cut the shuffled stream into mega-batches of MegaBatchMult * BatchSize,
  //   3. sort each mega-batch by sample length (descending), and
  //   4. (transformers detail) swap the single longest sample into the first
  //      mega-batch so the very first emitted batch contains the global longest
  //      sample (surfaces an out-of-memory early rather than mid-epoch),
  //   5. yield consecutive BatchSize chunks of the resulting index order.
  // A fresh Reseed reproduces the order bit-for-bit; the internal LCG is the
  // same one the sibling collators use, independent of the global RandSeed so
  // batching never perturbs weight-init randomness.
  //
  // The token-id convention matches the rest of the NLP pipeline (pad = 0 by
  // default; real corpus tokens carry their own ids). Each stored sample is one
  // variable-length token sequence; GetBatchPair emits a per-position causal-LM
  // training pair for the whole batch: every sample padded (on the right) to the
  // batch's BatchSeqLen, input = token ids, target = per-position one-hot of the
  // next token, with pad-target rows left all-zero for ApplyLossMask.
  // Coded by Claude (AI).
  TNNetLengthGroupedBatcher = class(TObject)
    private
      FSamples: array of TNeuralIntegerArray;
      FSampleCount: integer;
      FPadToken: integer;
      FVocabSize: integer;
      FBatchSize: integer;
      FMegaBatchMult: integer;
      FRngState: cardinal;
      FOrder: TNeuralIntegerArray;   // sample indices in emission order
      FBatchCount: integer;
      FIsBuilt: boolean;
      function NextRandom(): TNeuralFloat;
      function NextRandomInt(N: integer): integer;
      procedure RequireBuilt();
      procedure ShuffleOrder();
      procedure SortRangeByLenDesc(Lo, Hi: integer);
    public
      // pVocabSize: number of token ids (only used to size one-hot targets).
      // pBatchSize: samples per emitted batch. pMegaBatchMult: mega-batch size
      // is pMegaBatchMult * pBatchSize (transformers default 50); a larger
      // multiplier groups lengths more tightly but reduces shuffle randomness.
      constructor Create(pVocabSize: integer; pBatchSize: integer;
        pMegaBatchMult: integer = 50; pPadToken: integer = 0);
      destructor Destroy; override;
      // Removes all samples and any built batch order.
      procedure Clear();
      // Adds one variable-length token sequence (stored as-is; no separator is
      // appended). At least one token is required.
      procedure AddSample(const Tokens: array of integer);
      // Char-level convenience: sample tokens are Ord() of each character.
      procedure AddSampleFromString(const Str: string);
      // Reseeds the internal RNG for a reproducible megabatch shuffle.
      procedure Reseed(Seed: cardinal);
      // Builds the megabatch-shuffled emission order and the batch partition.
      // Call once per epoch (Reseed first for a different shuffle each epoch).
      procedure BuildBatches();
      function BatchCount(): integer;
      // Number of samples in batch BatchIdx (BatchSize, except possibly the
      // last batch when SampleCount is not a multiple of BatchSize).
      function BatchSize(BatchIdx: integer): integer;
      // Pad length of batch BatchIdx: the maximum sample length in that batch
      // (every sample in the batch is right-padded to this length).
      function BatchSeqLen(BatchIdx: integer): integer;
      // Original sample index of the WithinIdx-th member of batch BatchIdx.
      function SampleIndexOf(BatchIdx, WithinIdx: integer): integer;
      // Length (real token count, no padding) of the WithinIdx-th sample of
      // batch BatchIdx.
      function SampleLenOf(BatchIdx, WithinIdx: integer): integer;
      // Fills a per-position causal-LM training pair for ONE sample (the
      // WithinIdx-th member of batch BatchIdx), padded to the batch's
      // BatchSeqLen. Same volume layout as TNNetSequencePacker.GetTrainingPair:
      // pInput/pTarget have SizeX = BatchSeqLen, SizeY = 1. Input carries token
      // ids on the X axis when pInput.Depth = 1 (embedding pipelines) or one-hot
      // across the depth axis otherwise; pad positions hold the pad token.
      // Target: per-position one-hot of the next token within the sample, with
      // pad-target rows (everything from the sample's last real token onward)
      // left all-zero so a softmax head with ApplyLossMask trains only on the
      // real next-token targets.
      procedure GetTrainingPair(BatchIdx, WithinIdx: integer;
        pInput, pTarget: TNNetVolume);
      // Copies the network output into Desired at every non-predictable position
      // of the WithinIdx-th sample of batch BatchIdx (positions at/after the
      // sample's last real token), so e = Output - Desired is zero there (same
      // contract as TNNetSequencePacker.ApplyLossMask).
      procedure ApplyLossMask(BatchIdx, WithinIdx: integer;
        Desired, Actual: TNNetVolume);
      // Total pad tokens emitted across all batches with the current dynamic
      // (per-batch) padding: sum over batches of
      // (BatchSeqLen * BatchSize - real tokens in the batch).
      function TotalPadTokens(): int64;
      // Pad tokens that NAIVE global padding would emit on the same corpus:
      // (global max length) * SampleCount - total real tokens. The dynamic
      // batching above is guaranteed <= this (and strictly less whenever the
      // corpus has any length variation across batches).
      function NaiveTotalPadTokens(): int64;
      property SampleCount: integer read FSampleCount;
      property PadToken: integer read FPadToken;
      property VocabSize: integer read FVocabSize;
      property MegaBatchMult: integer read FMegaBatchMult;
  end;

implementation

uses
  math, neuralthread,
  {$IFDEF FPC}Classes,SysUtils,fileutil,fpjson,jsonparser{$ELSE}
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
  dir, Path, SearchPattern: {$IFDEF FPC} ShortString; {$ELSE} String; {$ENDIF}
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
  fileName, Path, SearchPattern: {$IFDEF FPC} ShortString; {$ELSE} String; {$ENDIF}
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
  CntHit{, CntMiss}: integer;
  InputString: string;
  MaxIdx: integer;
begin
  pInput := TNNetVolume.Create();
  pOutput := TNNetVolume.Create();
  CntHit := 0;
  //CntMiss := 0;
  // Make sure that expected input and output have the proper sizes.
  if NN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(NN.GetFirstLayer().Output);
  if NN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(NN.GetLastLayer().Output);
  MaxIdx := Samples - 1;
  for SampleId := 0 to MaxIdx do
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
    Move(Dataset[SampleId][0], AIntegerArray[0], SampleCutPosition * csIntegerSize);
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
      //Inc(CntMiss);
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
  MaxIdx: integer;
begin
  MaxIdx := Length(FDocs) - 1;
  for I := 0 to MaxIdx do SetLength(FDocs[I], 0);
  SetLength(FDocs, 0);
  FDocCount := 0;
  MaxIdx := Length(FWindows) - 1;
  for I := 0 to MaxIdx do SetLength(FWindows[I], 0);
  SetLength(FWindows, 0);
  FIsPacked := false;
end;

procedure TNNetSequencePacker.AddDocument(const Tokens: array of integer);
var
  I: integer;
  MaxIdx: integer;
begin
  MaxIdx := Length(Tokens) - 1;
  for I := 0 to MaxIdx do
  begin
    if Tokens[I] < 2 then
      raise Exception.Create(
        'TNNetSequencePacker: real token ids must be >= 2 ' +
        '(ids < 2 are reserved for pad/separator); got ' +
        IntToStr(Tokens[I]) + '.');
  end;
  if FDocCount >= Length(FDocs) then SetLength(FDocs, 8 + FDocCount * 2);
  SetLength(FDocs[FDocCount], Length(Tokens));
  if MaxIdx >= 0 then
    Move(Tokens[0], FDocs[FDocCount][0], Length(Tokens) * csIntegerSize);
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
  DocM1, WinM1, ContextM1: integer;
  DocLenM1: integer;
begin
  StreamLen := 0;
  DocM1 := FDocCount - 1;
  for DocIdx := 0 to DocM1 do
    StreamLen := StreamLen + Length(FDocs[DocIdx]) + 1; // +1 separator
  SetLength(Stream, StreamLen);
  Pos := 0;
  for DocIdx := 0 to DocM1 do
  begin
    DocLenM1 := Length(FDocs[DocIdx]) - 1;
    if DocLenM1 >= 0 then
    begin
      Move(FDocs[DocIdx][0], Stream[Pos], (DocLenM1 + 1) * csIntegerSize);
      Inc(Pos, DocLenM1 + 1);
    end;
    Stream[Pos] := FSeparatorToken;
    Inc(Pos);
  end;
  WinCount := (StreamLen + FContextLen - 1) div FContextLen;
  SetLength(FWindows, WinCount);
  WinM1 := WinCount - 1;
  ContextM1 := FContextLen - 1;
  for W := 0 to WinM1 do
  begin
    SetLength(FWindows[W], FContextLen);
    for I := 0 to ContextM1 do
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
  DocM1: integer;
  DocLenM1: integer;

  procedure PadAndCloseCurrent();
  var
    P: integer;
    ContextM1: integer;
  begin
    ContextM1 := FContextLen - 1;
    for P := Pos to ContextM1 do FWindows[W][P] := FPadToken;
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
  DocM1 := FDocCount - 1;
  for DocIdx := 0 to DocM1 do
  begin
    DocLen := Length(FDocs[DocIdx]);
    if DocLen > FContextLen - 1 then DocLen := FContextLen - 1;
    if (Pos + DocLen + 1 > FContextLen) or
      (OneDocPerWindow and (Pos > 0)) then
    begin
      if W >= 0 then PadAndCloseCurrent();
      OpenNewWindow();
    end;
    DocLenM1 := DocLen - 1;
    for I := 0 to DocLenM1 do
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
  ContextM1: integer;
begin
  RequirePacked();
  SetLength(Result, FContextLen);
  ContextM1 := FContextLen - 1;
  if ContextM1 >= 0 then
    Move(FWindows[WindowIdx][0], Result[0], FContextLen * csIntegerSize);
end;

function TNNetSequencePacker.GetToken(WindowIdx, Pos: integer): integer;
begin
  RequirePacked();
  Result := FWindows[WindowIdx][Pos];
end;

function TNNetSequencePacker.GetSegmentIds(WindowIdx: integer): TNeuralIntegerArray;
var
  Pos, SegId, PadId: integer;
  ContextM1: integer;
begin
  RequirePacked();
  SetLength(Result, FContextLen);
  // First pass: real/separator tokens get an incrementing document id; a new
  // document opens right after each separator. Pad positions are tagged -1 and
  // reassigned a single shared id below.
  SegId := 0;
  ContextM1 := FContextLen - 1;
  for Pos := 0 to ContextM1 do
  begin
    if FWindows[WindowIdx][Pos] = FPadToken then
    begin
      Result[Pos] := -1;
    end
    else
    begin
      Result[Pos] := SegId;
      if FWindows[WindowIdx][Pos] = FSeparatorToken then Inc(SegId);
    end;
  end;
  // Pads share one id distinct from every real-document id (= SegId, the next
  // unused value), so a pad never matches any real document.
  PadId := SegId;
  for Pos := 0 to ContextM1 do
    if Result[Pos] = -1 then Result[Pos] := PadId;
end;

procedure TNNetSequencePacker.GetSegmentVolume(WindowIdx: integer;
  pSegment: TNNetVolume);
var
  Ids: TNeuralIntegerArray;
  Pos: integer;
  ContextM1: integer;
begin
  RequirePacked();
  if (pSegment.SizeX <> FContextLen) or (pSegment.SizeY <> 1) or
     (pSegment.Depth <> 1) then
    raise Exception.Create('TNNetSequencePacker.GetSegmentVolume: pSegment ' +
      'must be (ContextLen,1,1). Got (' + IntToStr(pSegment.SizeX) + ',' +
      IntToStr(pSegment.SizeY) + ',' + IntToStr(pSegment.Depth) + ').');
  Ids := GetSegmentIds(WindowIdx);
  ContextM1 := FContextLen - 1;
  for Pos := 0 to ContextM1 do
    pSegment.FData[Pos] := Ids[Pos];
  SetLength(Ids, 0);
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
  ContextM2: integer;
begin
  Result := 0;
  ContextM2 := FContextLen - 2;
  for Pos := 0 to ContextM2 do
    if IsTargetPredictable(WindowIdx, Pos) then Inc(Result);
end;

function TNNetSequencePacker.Utilization(): TNeuralFloat;
var
  W, TotalSlots, Predictable: integer;
  WinM1: integer;
begin
  RequirePacked();
  Result := 0;
  TotalSlots := Length(FWindows) * (FContextLen - 1);
  if TotalSlots = 0 then exit;
  Predictable := 0;
  WinM1 := Length(FWindows) - 1;
  for W := 0 to WinM1 do
    Predictable := Predictable + PredictableTargetCount(W);
  Result := Predictable / TotalSlots;
end;

procedure TNNetSequencePacker.GetTrainingPair(WindowIdx: integer;
  pInput, pTarget: TNNetVolume);
var
  Pos, Token: integer;
  ContextM1, ContextM2, InputDepth: integer;
  IsIdInput: boolean;
begin
  RequirePacked();
  if pInput.SizeX <> FContextLen then
    raise Exception.Create('TNNetSequencePacker.GetTrainingPair: input SizeX ' +
      IntToStr(pInput.SizeX) + ' <> ContextLen ' + IntToStr(FContextLen) + '.');
  if pTarget.SizeX <> FContextLen then
    raise Exception.Create('TNNetSequencePacker.GetTrainingPair: target SizeX ' +
      IntToStr(pTarget.SizeX) + ' <> ContextLen ' + IntToStr(FContextLen) + '.');
  pInput.Fill(0);
  ContextM1 := FContextLen - 1;
  ContextM2 := FContextLen - 2;
  InputDepth := pInput.Depth;
  IsIdInput := (InputDepth = 1);
  for Pos := 0 to ContextM1 do
  begin
    Token := FWindows[WindowIdx][Pos];
    if IsIdInput
    then pInput.FData[Pos] := Token              // token ids -> embedding
    else if Token < InputDepth
    then pInput[Pos, 0, Token] := 1;             // one-hot across depth
  end;
  pTarget.Fill(0);
  for Pos := 0 to ContextM2 do
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
  Pos: integer;
  ContextM1, DepthM1, DesBase, ActBase, CopyBytes: integer;
begin
  RequirePacked();
  ContextM1 := FContextLen - 1;
  DepthM1 := Desired.Depth - 1;
  CopyBytes := (DepthM1 + 1) * csNeuralFloatSize;
  for Pos := 0 to ContextM1 do
  begin
    if not IsTargetPredictable(WindowIdx, Pos) then
    begin
      DesBase := Desired.GetRawPos(Pos, 0, 0);
      ActBase := Actual.GetRawPos(Pos, 0, 0);
      Move(Actual.FData[ActBase], Desired.FData[DesBase], CopyBytes);
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
  SpecialM1: integer;
begin
  Result := false;
  SpecialM1 := FSpecialCount - 1;
  for I := 0 to SpecialM1 do
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
  LenM1: integer;
begin
  Len := Length(Tokens);
  SetLength(CorruptedIds, Len);
  SetLength(Labels, Len);
  LenM1 := Len - 1;
  for I := 0 to LenM1 do
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
  LenM1: integer;
begin
  Len := Length(Tokens);
  if Length(WordIds) <> Len then
    raise Exception.Create(
      'TNNetMaskedLMCollator.CollateWholeWord: Tokens and WordIds lengths ' +
      'differ.');
  SetLength(CorruptedIds, Len);
  SetLength(Labels, Len);
  LenM1 := Len - 1;
  for I := 0 to LenM1 do
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
  LenM1: integer;
  IsIdInput: boolean;
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
  LenM1 := Len - 1;
  IsIdInput := (pInput.Depth = 1);
  for P := 0 to LenM1 do
  begin
    if IsIdInput then
      pInput.FData[P] := CorruptedIds[P]            // token ids on the X axis
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
  P: integer;
  LabelsM1, DepthM1, DesBase, ActBase, CopyBytes: integer;
begin
  LabelsM1 := Length(Labels) - 1;
  DepthM1 := Desired.Depth - 1;
  CopyBytes := (DepthM1 + 1) * csNeuralFloatSize;
  for P := 0 to LabelsM1 do
    if Labels[P] = csMaskedLMIgnoreLabel then
    begin
      DesBase := Desired.GetRawPos(P, 0, 0);
      ActBase := Actual.GetRawPos(P, 0, 0);
      Move(Actual.FData[ActBase], Desired.FData[DesBase], CopyBytes);
    end;
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
  SpecialM1: integer;
begin
  Result := false;
  SpecialM1 := FSpecialCount - 1;
  for I := 0 to SpecialM1 do
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
  LenM1: integer;
begin
  Len := Length(Tokens);
  NumSpans := 0;
  SetLength(Masked, Len);
  LenM1 := Len - 1;
  for I := 0 to LenM1 do Masked[I] := false;
  // Token budget to mask (~CorruptionRate of non-special tokens).
  Budget := 0;
  for I := 0 to LenM1 do
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
  SrcM1, TgtM1: integer;
  IsSrcIdInput, IsTgtIdInput: boolean;
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
  SrcM1 := Length(SourceIds) - 1;
  TgtM1 := Length(TargetIds) - 1;
  IsSrcIdInput := (pSource.Depth = 1);
  IsTgtIdInput := (pTarget.Depth = 1);
  for P := 0 to SrcM1 do
    if IsSrcIdInput then
      pSource.FData[P] := SourceIds[P]
    else
      pSource[P, 0, SourceIds[P]] := 1;
  for P := 0 to TgtM1 do
    if IsTgtIdInput then
      pTarget.FData[P] := TargetIds[P]
    else
      pTarget[P, 0, TargetIds[P]] := 1;
end;

{ TNNetLengthGroupedBatcher }

constructor TNNetLengthGroupedBatcher.Create(pVocabSize: integer;
  pBatchSize: integer; pMegaBatchMult: integer; pPadToken: integer);
begin
  inherited Create();
  if pVocabSize < 2 then
    raise Exception.Create('TNNetLengthGroupedBatcher: VocabSize must be >= 2.');
  if pBatchSize < 1 then
    raise Exception.Create('TNNetLengthGroupedBatcher: BatchSize must be >= 1.');
  if pMegaBatchMult < 1 then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher: MegaBatchMult must be >= 1.');
  FVocabSize := pVocabSize;
  FBatchSize := pBatchSize;
  FMegaBatchMult := pMegaBatchMult;
  FPadToken := pPadToken;
  FSampleCount := 0;
  FBatchCount := 0;
  FIsBuilt := false;
  FRngState := 314159265;
  SetLength(FSamples, 0);
  SetLength(FOrder, 0);
end;

destructor TNNetLengthGroupedBatcher.Destroy;
begin
  Clear();
  inherited Destroy;
end;

procedure TNNetLengthGroupedBatcher.Clear();
begin
  SetLength(FSamples, 0);
  SetLength(FOrder, 0);
  FSampleCount := 0;
  FBatchCount := 0;
  FIsBuilt := false;
end;

procedure TNNetLengthGroupedBatcher.AddSample(const Tokens: array of integer);
var
  I: integer;
  TokM1: integer;
begin
  if Length(Tokens) < 1 then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher.AddSample: a sample must have >= 1 token.');
  if FSampleCount >= Length(FSamples) then
    SetLength(FSamples, (FSampleCount + 1) * 2);
  SetLength(FSamples[FSampleCount], Length(Tokens));
  Move(Tokens[0], FSamples[FSampleCount][0], Length(Tokens) * csIntegerSize);
  Inc(FSampleCount);
  FIsBuilt := false;
end;

procedure TNNetLengthGroupedBatcher.AddSampleFromString(const Str: string);
var
  Tokens: TNeuralIntegerArray;
  I: integer;
  StrLen: integer;
begin
  SetLength(Tokens, Length(Str));
  StrLen := Length(Str);
  for I := 1 to StrLen do
    Tokens[I - 1] := Ord(Str[I]);
  AddSample(Tokens);
end;

procedure TNNetLengthGroupedBatcher.Reseed(Seed: cardinal);
begin
  FRngState := Seed;
  FIsBuilt := false;
end;

function TNNetLengthGroupedBatcher.NextRandom(): TNeuralFloat;
begin
  // Same Numerical Recipes LCG as the sibling collators; independent of the
  // global RandSeed so the shuffle never perturbs weight-init randomness.
  FRngState := FRngState * 1664525 + 1013904223;
  Result := (FRngState shr 8) / 16777216.0; // top 24 bits / 2^24
end;

function TNNetLengthGroupedBatcher.NextRandomInt(N: integer): integer;
begin
  if N <= 1 then begin Result := 0; exit; end;
  Result := Trunc(NextRandom() * N);
  if Result >= N then Result := N - 1;
end;

procedure TNNetLengthGroupedBatcher.RequireBuilt();
begin
  if not FIsBuilt then
    raise Exception.Create('TNNetLengthGroupedBatcher: call BuildBatches first.');
end;

procedure TNNetLengthGroupedBatcher.ShuffleOrder();
var
  I, J, Tmp: integer;
  SampleM1: integer;
begin
  // Fisher-Yates over the sample indices using the internal LCG.
  SampleM1 := FSampleCount - 1;
  for I := 0 to SampleM1 do FOrder[I] := I;
  for I := FSampleCount - 1 downto 1 do
  begin
    J := NextRandomInt(I + 1);
    Tmp := FOrder[I]; FOrder[I] := FOrder[J]; FOrder[J] := Tmp;
  end;
end;

procedure TNNetLengthGroupedBatcher.SortRangeByLenDesc(Lo, Hi: integer);
var
  I, J, PivotLen, Tmp: integer;
begin
  // Plain quicksort on FOrder[Lo..Hi] keyed by descending sample length.
  if Lo >= Hi then exit;
  PivotLen := Length(FSamples[FOrder[(Lo + Hi) div 2]]);
  I := Lo; J := Hi;
  repeat
    while Length(FSamples[FOrder[I]]) > PivotLen do Inc(I);
    while Length(FSamples[FOrder[J]]) < PivotLen do Dec(J);
    if I <= J then
    begin
      Tmp := FOrder[I]; FOrder[I] := FOrder[J]; FOrder[J] := Tmp;
      Inc(I); Dec(J);
    end;
  until I > J;
  SortRangeByLenDesc(Lo, J);
  SortRangeByLenDesc(I, Hi);
end;

procedure TNNetLengthGroupedBatcher.BuildBatches();
var
  Mega, Lo, Hi, I, LongestPos, Tmp: integer;
  SampleM1: integer;
begin
  if FSampleCount < 1 then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher.BuildBatches: no samples added.');
  SetLength(FOrder, FSampleCount);
  // 1. shuffle all indices.
  ShuffleOrder();
  // 2-3. cut into mega-batches and sort each by length (descending).
  Mega := FMegaBatchMult * FBatchSize;
  Lo := 0;
  while Lo < FSampleCount do
  begin
    Hi := Lo + Mega - 1;
    if Hi > FSampleCount - 1 then Hi := FSampleCount - 1;
    SortRangeByLenDesc(Lo, Hi);
    Lo := Lo + Mega;
  end;
  // 4. transformers detail: ensure the GLOBAL longest sample sits in the first
  //    mega-batch (now at FOrder[0], since each mega-batch is length-sorted) so
  //    the very first emitted batch surfaces a worst-case OOM immediately.
  if FSampleCount > 1 then
  begin
    LongestPos := 0;
    SampleM1 := FSampleCount - 1;
    for I := 1 to SampleM1 do
      if Length(FSamples[FOrder[I]]) > Length(FSamples[FOrder[LongestPos]]) then
        LongestPos := I;
    if LongestPos <> 0 then
    begin
      Tmp := FOrder[0]; FOrder[0] := FOrder[LongestPos]; FOrder[LongestPos] := Tmp;
    end;
  end;
  // 5. partition into BatchSize chunks.
  FBatchCount := (FSampleCount + FBatchSize - 1) div FBatchSize;
  FIsBuilt := true;
end;

function TNNetLengthGroupedBatcher.BatchCount(): integer;
begin
  RequireBuilt();
  Result := FBatchCount;
end;

function TNNetLengthGroupedBatcher.BatchSize(BatchIdx: integer): integer;
begin
  RequireBuilt();
  if (BatchIdx < 0) or (BatchIdx >= FBatchCount) then
    raise Exception.Create('TNNetLengthGroupedBatcher.BatchSize: bad batch index.');
  Result := FBatchSize;
  if (BatchIdx = FBatchCount - 1) and (FSampleCount mod FBatchSize <> 0) then
    Result := FSampleCount mod FBatchSize;
end;

function TNNetLengthGroupedBatcher.SampleIndexOf(BatchIdx, WithinIdx: integer): integer;
var
  Flat: integer;
begin
  RequireBuilt();
  if (WithinIdx < 0) or (WithinIdx >= BatchSize(BatchIdx)) then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher.SampleIndexOf: bad within-batch index.');
  Flat := BatchIdx * FBatchSize + WithinIdx;
  Result := FOrder[Flat];
end;

function TNNetLengthGroupedBatcher.SampleLenOf(BatchIdx, WithinIdx: integer): integer;
begin
  Result := Length(FSamples[SampleIndexOf(BatchIdx, WithinIdx)]);
end;

function TNNetLengthGroupedBatcher.BatchSeqLen(BatchIdx: integer): integer;
var
  W, L: integer;
  MaxW: integer;
begin
  RequireBuilt();
  Result := 0;
  MaxW := BatchSize(BatchIdx) - 1;
  for W := 0 to MaxW do
  begin
    L := SampleLenOf(BatchIdx, W);
    if L > Result then Result := L;
  end;
end;

procedure TNNetLengthGroupedBatcher.GetTrainingPair(BatchIdx, WithinIdx: integer;
  pInput, pTarget: TNNetVolume);
var
  Sample: TNeuralIntegerArray;
  SeqLen, Len, Pos, Token: integer;
  SeqM1, LenM2: integer;
  IsIdInput: boolean;
begin
  RequireBuilt();
  SeqLen := BatchSeqLen(BatchIdx);
  if pInput.SizeX < SeqLen then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher.GetTrainingPair: input SizeX (' +
      IntToStr(pInput.SizeX) + ') < batch seq len (' + IntToStr(SeqLen) + ').');
  if pTarget.SizeX < SeqLen then
    raise Exception.Create(
      'TNNetLengthGroupedBatcher.GetTrainingPair: target SizeX (' +
      IntToStr(pTarget.SizeX) + ') < batch seq len (' + IntToStr(SeqLen) + ').');
  Sample := FSamples[SampleIndexOf(BatchIdx, WithinIdx)];
  Len := Length(Sample);
  // Input: real tokens then right-padding to the batch's seq len.
  pInput.Fill(0);
  SeqM1 := SeqLen - 1;
  LenM2 := Len - 2;
  IsIdInput := (pInput.Depth = 1);
  for Pos := 0 to SeqM1 do
  begin
    if Pos < Len then Token := Sample[Pos] else Token := FPadToken;
    if IsIdInput
    then pInput.FData[Pos] := Token
    else if (Token >= 0) and (Token < pInput.Depth)
    then pInput[Pos, 0, Token] := 1;
  end;
  // Target: per-position one-hot of the NEXT real token (positions 0..Len-2);
  // every padded position and the sample's last real token carry no target.
  pTarget.Fill(0);
  for Pos := 0 to LenM2 do
  begin
    Token := Sample[Pos + 1];
    if (Token >= 0) and (Token < pTarget.Depth) then pTarget[Pos, 0, Token] := 1;
  end;
end;

procedure TNNetLengthGroupedBatcher.ApplyLossMask(BatchIdx, WithinIdx: integer;
  Desired, Actual: TNNetVolume);
var
  SeqLen, Len, Pos: integer;
  SeqM1, DepthM1, DesBase, ActBase, CopyBytes: integer;
begin
  RequireBuilt();
  SeqLen := BatchSeqLen(BatchIdx);
  Len := SampleLenOf(BatchIdx, WithinIdx);
  SeqM1 := SeqLen - 1;
  DepthM1 := Desired.Depth - 1;
  CopyBytes := (DepthM1 + 1) * csNeuralFloatSize;
  for Pos := 0 to SeqM1 do
  begin
    // Predictable iff a next real token exists: Pos in 0..Len-2.
    if Pos > Len - 2 then
    begin
      DesBase := Desired.GetRawPos(Pos, 0, 0);
      ActBase := Actual.GetRawPos(Pos, 0, 0);
      Move(Actual.FData[ActBase], Desired.FData[DesBase], CopyBytes);
    end;
  end;
end;

function TNNetLengthGroupedBatcher.TotalPadTokens(): int64;
var
  B, W, SeqLen: integer;
  BatchM1: integer;
  BatchSizeM1: integer;
begin
  RequireBuilt();
  Result := 0;
  BatchM1 := FBatchCount - 1;
  for B := 0 to BatchM1 do
  begin
    SeqLen := BatchSeqLen(B);
    BatchSizeM1 := BatchSize(B) - 1;
    for W := 0 to BatchSizeM1 do
      Result := Result + (SeqLen - SampleLenOf(B, W));
  end;
end;

function TNNetLengthGroupedBatcher.NaiveTotalPadTokens(): int64;
var
  I, GlobalMax, L: integer;
  RealTokens: int64;
  SampleM1: integer;
begin
  GlobalMax := 0;
  RealTokens := 0;
  SampleM1 := FSampleCount - 1;
  for I := 0 to SampleM1 do
  begin
    L := Length(FSamples[I]);
    if L > GlobalMax then GlobalMax := L;
    RealTokens := RealTokens + L;
  end;
  Result := int64(GlobalMax) * FSampleCount - RealTokens;
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
  MaxIdx: integer;
begin
  Result := 0;
  if Count > 0 then
  begin
    MaxIdx := Count - 1;
    for ClassId := 0 to MaxIdx do
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
  //SourceVolume: TNNetVolume;
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

  for CountY := 0 to MaxY do
  begin
    RawPos := Vol.GetRawPos(0, CountY, 0);
    for CountX := 0 to MaxX do
    begin
      LocalColor := M.Colors[CountX, CountY];

      Vol.FData[RawPos]     := LocalColor.red shr 8;
      Vol.FData[RawPos + 1] := LocalColor.green shr 8;
      Vol.FData[RawPos + 2] := LocalColor.blue shr 8;
      Inc(RawPos, 3);
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
  for CountY := 0 to MaxY do
  begin
    RawPos := Vol.GetRawPos(0, CountY, 0);
    for CountX := 0 to MaxX do
    begin
      LocalColor.red := NeuronForceMinMax(Round(Vol.FData[RawPos]),0,255) shl 8;
      LocalColor.green := NeuronForceMinMax(Round(Vol.FData[RawPos + 1]),0,255) shl 8;
      LocalColor.blue := NeuronForceMinMax(Round(Vol.FData[RawPos + 2]),0, 255) shl 8;
      M.Colors[CountX, CountY] := LocalColor;
      Inc(RawPos, 3);
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

  for CountY := 0 to MaxY do
  begin
    RawPos := Vol.GetRawPos(0, CountY, 0);
    for CountX := 0 to MaxX do
    begin
      LocalColor := Picture.Bitmap.Canvas.Pixels[CountX, CountY];

      Vol.FData[RawPos]     := LocalColor and 255;
      Vol.FData[RawPos + 1] := (LocalColor shr 8) and 255;
      Vol.FData[RawPos + 2] := (LocalColor shr 16) and 255;
      Inc(RawPos, 3);
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
  DownLimit: integer;
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
    DownLimit := LastElement-(10000-ValidationSampleSize)+1;
    for I := LastElement downto DownLimit do
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
  DownLimit: integer;
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
    DownLimit := LastElement-ValidationSampleSize+1;
    for I := LastElement downto DownLimit do
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
  Vol: TNNetVolume;
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
    Vol := ImgVolumes[ImgPos];
    LoadTinyImageIntoNNetVolume(Img, Vol);

    if (color_encoding = csEncodeGray) then
    begin
      AuxVolume.Copy(Vol);
      Vol.GetGrayFromRgb(AuxVolume);
    end;

    Vol.RgbImgToNeuronalInput(color_encoding);

    Vol.GetMinMaxAtDepth(0, pMin, pMax); //WriteLn  (I:8,' - #0 Min:',pMin, ' Max:',pMax);

    globalMin0 := Math.Min(pMin, globalMin0);
    globalMax0 := Math.Max(pMax, globalMax0);

    if (Vol.Depth >= 2) then
    begin
      Vol.GetMinMaxAtDepth(1, pMin, pMax); //Write  (' #1 Min:',pMin, ' Max:',pMax);

      globalMin1 := Math.Min(pMin, globalMin1);
      globalMax1 := Math.Max(pMax, globalMax1);
    end;

    if (Vol.Depth >= 3) then
    begin
      Vol.GetMinMaxAtDepth(2, pMin, pMax); //WriteLn(' #2 Min:',pMin, ' Max:',pMax);

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

procedure PreprocessImageForVisionModel(Src, Dst: TNNetVolume;
  ResizeShorterSide, CropSize: integer;
  const Mean, Std: array of TNeuralFloat);
var
  Work: TNNetVolume;
  ResizeW, ResizeH, OffX, OffY, X, Y, C, SrcX, SrcY: integer;
  Scale, Fx, Fy, V: TNeuralFloat;
  ResizeWM1, ResizeHM1, CropM1, WorkBase, SrcBase, DstBase: integer;
  ScaleX, ScaleY: TNeuralFloat;
  SrcXMax, SrcYMax, DstStride, WorkStride: integer;
  WorkSizeXM1, WorkSizeYM1, WorkRowBase: integer;
  RowOutY: boolean;
begin
  if (Src = nil) or (Dst = nil) then
    raise Exception.Create('PreprocessImageForVisionModel: nil volume.');
  if Src = Dst then
    raise Exception.Create(
      'PreprocessImageForVisionModel: Src and Dst must differ.');
  if Src.Depth <> 3 then
    raise Exception.Create(
      'PreprocessImageForVisionModel: source must have depth 3 (RGB), got ' +
      IntToStr(Src.Depth) + '.');
  if (Length(Mean) < 3) or (Length(Std) < 3) then
    raise Exception.Create(
      'PreprocessImageForVisionModel: Mean/Std need 3 elements.');

  Work := TNNetVolume.Create;
  try
    // ---- (1) resize: shorter edge -> ResizeShorterSide, aspect preserved.
    // PIL/torchvision default is bicubic; this is bilinear (identity when the
    // source is already at the target size, so a pre-resized Src is exact).
    if (Src.SizeX <> ResizeShorterSide) or (Src.SizeY <> ResizeShorterSide) then
    begin
      if Src.SizeX <= Src.SizeY then
      begin
        Scale := ResizeShorterSide / Src.SizeX;
        ResizeW := ResizeShorterSide;
        ResizeH := Round(Src.SizeY * Scale);
      end
      else
      begin
        Scale := ResizeShorterSide / Src.SizeY;
        ResizeH := ResizeShorterSide;
        ResizeW := Round(Src.SizeX * Scale);
      end;
      if ResizeW < 1 then ResizeW := 1;
      if ResizeH < 1 then ResizeH := 1;
      Work.ReSize(ResizeW, ResizeH, 3);
      ResizeHM1 := ResizeH - 1;
      ResizeWM1 := ResizeW - 1;
      ScaleX := Src.SizeX / ResizeW;
      ScaleY := Src.SizeY / ResizeH;
      SrcXMax := Src.SizeX - 1;
      SrcYMax := Src.SizeY - 1;
      for Y := 0 to ResizeHM1 do
      begin
        // bilinear sample location (align_corners = false convention); Y-only
        Fy := (Y + 0.5) * ScaleY - 0.5;
        if Fy < 0 then Fy := 0;
        SrcY := Trunc(Fy);
        if SrcY > SrcYMax then SrcY := SrcYMax;
        for X := 0 to ResizeWM1 do
        begin
          Fx := (X + 0.5) * ScaleX - 0.5;
          if Fx < 0 then Fx := 0;
          SrcX := Trunc(Fx);
          if SrcX > SrcXMax then SrcX := SrcXMax;
          WorkBase := Work.GetRawPos(X, Y, 0);
          SrcBase := Src.GetRawPos(SrcX, SrcY, 0);
          Move(Src.FData[SrcBase], Work.FData[WorkBase], 3 * csNeuralFloatSize);
        end;
      end;
    end
    else
      Work.Copy(Src);

    // ---- (2) center crop to (CropSize, CropSize).
    OffX := (Work.SizeX - CropSize) div 2;
    OffY := (Work.SizeY - CropSize) div 2;
    Dst.ReSize(CropSize, CropSize, 3);
    CropM1 := CropSize - 1;
    DstStride := Dst.GetRawPos(1, 0, 0);
    WorkStride := Work.GetRawPos(1, 0, 0);
    WorkSizeXM1 := Work.SizeX - 1;
    WorkSizeYM1 := Work.SizeY - 1;
    for Y := 0 to CropM1 do
    begin
      SrcY := OffY + Y;
      RowOutY := (SrcY < 0) or (SrcY > WorkSizeYM1);
      if not RowOutY then WorkRowBase := Work.GetRawPos(0, SrcY, 0)
      else WorkRowBase := 0;
      DstBase := Dst.GetRawPos(0, Y, 0);
      for X := 0 to CropM1 do
      begin
        SrcX := OffX + X;
        // pad with zeros if the crop window exceeds the resized image
        if RowOutY or (SrcX < 0) or (SrcX > WorkSizeXM1) then
        begin
          for C := 0 to 2 do Dst.FData[DstBase + C] := 0;
        end
        else
        begin
          // ---- (3) rescale by 1/255 then per-channel normalize.
          WorkBase := WorkRowBase + SrcX * WorkStride;
          for C := 0 to 2 do
          begin
            V := Work.FData[WorkBase + C] / 255.0;
            Dst.FData[DstBase + C] := (V - Mean[C]) / Std[C];
          end;
        end;
        Inc(DstBase, DstStride);
      end;
    end;
  finally
    Work.Free;
  end;
end;

function LoadImageForVisionModel(const ImageFileName: string; V: TNNetVolume;
  ResizeShorterSide, CropSize: integer;
  const Mean, Std: array of TNeuralFloat): boolean;
var
  Raw: TNNetVolume;
begin
  Raw := TNNetVolume.Create;
  try
    Result := LoadImageFromFileIntoVolume(ImageFileName, Raw);
    if Result then
      PreprocessImageForVisionModel(Raw, V, ResizeShorterSide, CropSize,
        Mean, Std);
  finally
    Raw.Free;
  end;
end;

{$IFDEF FPC}
function ReadImagePreprocessConfig(
  const FileName: string): TNNetImagePreprocess;
var
  JsonText: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;

  // Reads a size-like field that may be an int (square) or an object with
  // shortest_edge / height / width. Returns the resolved edge length.
  function ReadEdge(const FieldName: string; DefaultEdge: integer): integer;
  var
    Data: TJSONData;
    O: TJSONObject;
  begin
    Result := DefaultEdge;
    Data := Obj.Find(FieldName);
    if Data = nil then Exit;
    if Data is TJSONObject then
    begin
      O := TJSONObject(Data);
      if O.IndexOfName('shortest_edge') >= 0 then
        Result := O.Get('shortest_edge', DefaultEdge)
      else if O.IndexOfName('height') >= 0 then
        Result := O.Get('height', DefaultEdge)
      else if O.IndexOfName('width') >= 0 then
        Result := O.Get('width', DefaultEdge);
    end
    else if Data.JSONType = jtNumber then
      Result := Data.AsInteger;
  end;

  procedure ReadTriple(const FieldName: string; var Arr: array of TNeuralFloat;
    D0, D1, D2: TNeuralFloat);
  var
    Data: TJSONData;
    A: TJSONArray;
  begin
    Arr[0] := D0; Arr[1] := D1; Arr[2] := D2;
    Data := Obj.Find(FieldName);
    if (Data <> nil) and (Data is TJSONArray) then
    begin
      A := TJSONArray(Data);
      if A.Count >= 3 then
      begin
        Arr[0] := A.Items[0].AsFloat;
        Arr[1] := A.Items[1].AsFloat;
        Arr[2] := A.Items[2].AsFloat;
      end;
    end;
  end;

begin
  if not FileExists(FileName) then
    raise Exception.Create(
      'ReadImagePreprocessConfig: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    try
      Root := GetJSON(JsonText.Text);
    except
      on E: Exception do
        raise Exception.Create('ReadImagePreprocessConfig: config "' +
          FileName + '" is not valid JSON (' + E.Message + ').');
    end;
    if not (Root is TJSONObject) then
      raise Exception.Create('ReadImagePreprocessConfig: config "' +
        FileName + '" is not a JSON object.');
    Obj := TJSONObject(Root);

    Result.ResizeShorterSide := ReadEdge('size', 224);
    Result.CropSize := ReadEdge('crop_size', Result.ResizeShorterSide);
    // Defaults follow CLIPImageProcessor (the OpenAI image_mean/image_std).
    ReadTriple('image_mean', Result.Mean,
      csClipMean[0], csClipMean[1], csClipMean[2]);
    ReadTriple('image_std', Result.Std,
      csClipStd[0], csClipStd[1], csClipStd[2]);
  finally
    Root.Free;
    JsonText.Free;
  end;
end;
{$ELSE}
function ReadImagePreprocessConfig(
  const FileName: string): TNNetImagePreprocess;
var
  JsonText: TStringList;
  Root: TJSONValue;
  Obj: TJSONObject;

  // Reads a size-like field that may be an int (square) or an object with
  // shortest_edge / height / width. Returns the resolved edge length.
  function ReadEdge(const FieldName: string; DefaultEdge: integer): integer;
  var
    Data: TJSONValue;
    O: TJSONObject;
    Temp: TJSONValue;
  begin
    Result := DefaultEdge;
    Data := Obj.FindValue(FieldName);
    if Data = nil then Exit;

    if Data is TJSONObject then
    begin
      O := Data as TJSONObject;
      if O.TryGetValue('shortest_edge', Temp) then
        Result := (Temp as TJSONNumber).AsInt
      else if O.TryGetValue('height', Temp) then
        Result := (Temp as TJSONNumber).AsInt
      else if O.TryGetValue('width', Temp) then
        Result := (Temp as TJSONNumber).AsInt;
    end
    else if Data is TJSONNumber then
      Result := (Data as TJSONNumber).AsInt;
  end;

  procedure ReadTriple(const FieldName: string; var Arr: array of TNeuralFloat;
    D0, D1, D2: TNeuralFloat);
  var
    Data: TJSONValue;
    A: TJSONArray;
  begin
    Arr[0] := D0; Arr[1] := D1; Arr[2] := D2;
    Data := Obj.FindValue(FieldName);
    if (Data <> nil) and (Data is TJSONArray) then
    begin
      A := Data as TJSONArray;
      if A.Count >= 3 then
      begin
        Arr[0] := (A.Items[0] as TJSONNumber).AsDouble;
        Arr[1] := (A.Items[1] as TJSONNumber).AsDouble;
        Arr[2] := (A.Items[2] as TJSONNumber).AsDouble;
      end;
    end;
  end;

begin
  if not FileExists(FileName) then
    raise Exception.Create(
      'ReadImagePreprocessConfig: config file not found: ' + FileName);
  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(FileName);
    Root := TJSONObject.ParseJSONValue(JsonText.Text);
    if Root = nil then
      raise Exception.Create('ReadImagePreprocessConfig: config "' +
        FileName + '" is not valid JSON.');
    if not (Root is TJSONObject) then
      raise Exception.Create('ReadImagePreprocessConfig: config "' +
        FileName + '" is not a JSON object.');
    Obj := Root as TJSONObject;

    Result.ResizeShorterSide := ReadEdge('size', 224);
    Result.CropSize := ReadEdge('crop_size', Result.ResizeShorterSide);
    // Defaults follow CLIPImageProcessor (the OpenAI image_mean/image_std).
    ReadTriple('image_mean', Result.Mean,
      csClipMean[0], csClipMean[1], csClipMean[2]);
    ReadTriple('image_std', Result.Std,
      csClipStd[0], csClipStd[1], csClipStd[2]);
  finally
    Root.Free;
    JsonText.Free;
  end;
end;
{$ENDIF}

procedure ConfusionWriteCSVHeader(var CSVConfusion: TextFile; Labels: array of string);
var
  I: integer;
  Hi: integer;
begin
  Hi := High(Labels);
  for I := Low(Labels) to Hi do
  begin
    if I > 0 then Write(CSVConfusion, ',');
    Write(CSVConfusion, Labels[I]);
  end;
  WriteLn(CSVConfusion);
end;

procedure ConfusionWriteCSV(var CSVConfusion: TextFile; Vol: TNNetVolume; Digits: integer);
var
  I, J: integer;
  SizeYM1, DepthM1, RowBase: integer;
begin
  SizeYM1 := Vol.SizeY - 1;
  DepthM1 := Vol.Depth - 1;
  for I := 0 to SizeYM1 do
  begin
    RowBase := Vol.GetRawPos(0, I, 0);
    for J := 0 to DepthM1 do
    begin
      if J > 0 then Write(CSVConfusion, ',');
      Write(CSVConfusion, Round(Vol.FData[RowBase + J]):Digits);
    end;
    WriteLn(CSVConfusion);
  end;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TTinyImage; Vol: TNNetVolume);
var
  I, J: integer;
  Pos, Stride: integer;
begin
  Vol.ReSize(32,32,3);
  Stride := Vol.GetRawPos(1, 0, 0);
  for I := 0 to 31 do
  begin
    Pos := Vol.GetRawPos(0, I, 0);
    for J := 0 to 31 do
    begin
      Vol.FData[Pos]     := TI.R[I, J];
      Vol.FData[Pos + 1] := TI.G[I, J];
      Vol.FData[Pos + 2] := TI.B[I, J];
      Inc(Pos, Stride);
    end;
  end;
  Vol.Tag := TI.bLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TCifar100Image; Vol: TNNetVolume);
var
  I, J: integer;
  Pos, Stride: integer;
begin
  Vol.ReSize(32,32,3);
  Stride := Vol.GetRawPos(1, 0, 0);
  for I := 0 to 31 do
  begin
    Pos := Vol.GetRawPos(0, I, 0);
    for J := 0 to 31 do
    begin
      Vol.FData[Pos]     := TI.R[I, J];
      Vol.FData[Pos + 1] := TI.G[I, J];
      Vol.FData[Pos + 2] := TI.B[I, J];
      Inc(Pos, Stride);
    end;
  end;
  Vol.Tags[0] := TI.bFineLabel;
  Vol.Tags[1] := TI.bCoarseLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TMNistImage; Vol: TNNetVolume);
var
  I, J: integer;
  Pos, Stride: integer;
begin
  Vol.ReSize(28, 28, 1);
  Stride := Vol.GetRawPos(1, 0, 0);
  for I := 0 to 27 do
  begin
    Pos := Vol.GetRawPos(0, I, 0);
    for J := 0 to 27 do
    begin
      Vol.FData[Pos] := TI[I, J];
      Inc(Pos, Stride);
    end;
  end;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TTinyImage);
var
  I, J: integer;
  Pos, Stride: integer;
begin
  Stride := Vol.GetRawPos(1, 0, 0);
  for I := 0 to 31 do
  begin
    Pos := Vol.GetRawPos(0, I, 0);
    for J := 0 to 31 do
    begin
      TI.R[I, J] := RoundAsByte(Vol.FData[Pos]);
      TI.G[I, J] := RoundAsByte(Vol.FData[Pos + 1]);
      TI.B[I, J] := RoundAsByte(Vol.FData[Pos + 2]);
      Inc(Pos, Stride);
    end;
  end;
  TI.bLabel := Vol.Tag;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TCifar100Image);
var
  I, J: integer;
  Pos, Stride: integer;
begin
  Stride := Vol.GetRawPos(1, 0, 0);
  for I := 0 to 31 do
  begin
    Pos := Vol.GetRawPos(0, I, 0);
    for J := 0 to 31 do
    begin
      TI.R[I, J] := RoundAsByte(Vol.FData[Pos]);
      TI.G[I, J] := RoundAsByte(Vol.FData[Pos + 1]);
      TI.B[I, J] := RoundAsByte(Vol.FData[Pos + 2]);
      Inc(Pos, Stride);
    end;
  end;
  TI.bCoarseLabel := Vol.Tags[0];
  TI.bFineLabel := Vol.Tags[1];
end;

procedure LoadTinySingleChannelIntoNNetVolume(var SC: TTinySingleChannelImage;
  Vol: TNNetVolume);
var
  I, J: integer;
  Pos, YStride: integer;
begin
  Vol.ReSize(32,32,1);
  Vol.Tag := SC.bLabel;
  YStride := Vol.GetRawPos(0, 1, 0);
  for I := 0 to 31 do
  begin
    Pos := Vol.GetRawPos(I, 0, 0);
    for J := 0 to 31 do
    begin
      Vol.FData[Pos] := SC.Grey[I,J];
      Inc(Pos, YStride);
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
  LocalImg: TNNetVolume;
  LocalTag: integer;
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

    LocalImg := ImgVolumes[ImgIdx];

    NN.Compute(LocalImg);
    NN.GetOutput(pOutput);

    LocalImg.FlipX();
    NN.AddOutput(pOutput);

    LocalTag := LocalImg.Tag;
    if pOutput.GetClass() = LocalTag then
    begin
      Inc(Hit);
    end
    else
    begin
      Inc(Miss);
    end;

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( LocalTag );
    end
    else
    begin
      vOutput.SetClassForReLU( LocalTag );
    end;

    ErrorSum := ErrorSum + vOutput.SumDiff(pOutput);

    if (bIsSoftmax) then
    begin
      OutputValue := pOutput.FData[ LocalTag ];
      if (OutputValue > 0) then
      begin
        CurrentLoss := -pcr_logf(OutputValue);
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
  MaxValue, RowCnt, WordCnt, SepCount: integer;
  Separator: TNNetStringList;
  SepCountM1: integer;
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
      SepCount := Separator.Count;
      SepCountM1 := SepCount - 1;
      for WordCnt := 0 to SepCountM1 do
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
  RowCnt, WordCnt, SepCount: integer;
  Separator: TNNetStringList;
  SepCountM1: integer;
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
    SepCount := Separator.Count;
    SetLength(aTokens[RowCnt], SepCount);
    if SepCount > 0 then
    begin
      SepCountM1 := SepCount - 1;
      for WordCnt := 0 to SepCountM1 do
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

// ----------------------------------------------------------------------------
//  RandAugment / TrivialAugment automatic augmentation policy - implementation
// ----------------------------------------------------------------------------

// Neuronal domain <-> 0..255 pixel helpers. RGB neuronal input uses
// p = (px - 128)/64, so px = p*64 + 128 and the valid neuronal range is
// [-2 .. (255-128)/64 = 1.984375]. We treat [-2..2] as the clamp domain to
// match the rest of the library (RgbImgToNeuronalInput comment).
const
  cAugNeuronMin: TNeuralFloat = -2.0;
  cAugNeuronMax: TNeuralFloat =  2.0;

function AugNeuronToPixel(v: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  Result := v * 64.0 + 128.0;
end;

function AugPixelToNeuron(p: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  Result := (p - 128.0) / 64.0;
end;

function AugClampNeuron(v: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  if v < cAugNeuronMin then Result := cAugNeuronMin
  else if v > cAugNeuronMax then Result := cAugNeuronMax
  else Result := v;
end;

function AugClampPixel(p: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  if p < 0 then Result := 0
  else if p > 255 then Result := 255
  else Result := p;
end;

// Linearly interpolate every pixel toward a per-channel "degenerate" image:
//   out = blend*degenerate + (1-blend)*original   (torchvision _blend).
// Here we blend toward a scalar gray value per the photometric op.
procedure AugBlendTowardScalar(V: TNNetVolume; GrayPixel, Factor: TNeuralFloat);
var
  I: integer;
  p, kAdd: TNeuralFloat;
  SizeM1: integer;
begin
  // out_px = Factor*orig_px + (1-Factor)*gray  (torchvision uses
  //   img2 = (1-ratio)*degenerate + ratio*img). Factor>1 enhances.
  SizeM1 := V.Size - 1;
  kAdd := (1.0 - Factor) * GrayPixel;
  for I := 0 to SizeM1 do
  begin
    p := AugNeuronToPixel(V.FData[I]);
    p := Factor * p + kAdd;
    V.FData[I] := AugClampNeuron(AugPixelToNeuron(AugClampPixel(p)));
  end;
end;

// Bilinear-free nearest geometric warp helper. Maps each destination pixel
// (dx,dy) back to a source pixel via an affine transform around the image
// center; out-of-bounds samples are filled with neutral gray (neuronal 0).
// Mat is [a,b,c, d,e,f] s.t. src = Mat * (dst_centered) + center.
procedure AugAffineWarp(V: TNNetVolume; const Mat: array of TNeuralFloat);
var
  W, H, Dep, dx, dy, d, sx, sy: integer;
  cx, cy, ox, oy, fx, fy, rowFx, rowFy: TNeuralFloat;
  Src: TNNetVolume;
  WM1, HM1, DepM1, VBase, SrcBase: integer;
begin
  W := V.SizeX; H := V.SizeY; Dep := V.Depth;
  if (W <= 0) or (H <= 0) then Exit;
  Src := TNNetVolume.Create;
  Src.Copy(V);
  cx := (W - 1) / 2.0;
  cy := (H - 1) / 2.0;
  WM1 := W - 1; HM1 := H - 1; DepM1 := Dep - 1;
  for dy := 0 to HM1 do
  begin
    oy := dy - cy;
    rowFx := Mat[1] * oy + Mat[2] + cx;
    rowFy := Mat[4] * oy + Mat[5] + cy;
    VBase := V.GetRawPos(0, dy, 0);
    for dx := 0 to WM1 do
    begin
      ox := dx - cx;
      fx := Mat[0] * ox + rowFx;
      fy := Mat[3] * ox + rowFy;
      sx := Round(fx);
      sy := Round(fy);
      if (sx >= 0) and (sx < W) and (sy >= 0) and (sy < H) then
      begin
        SrcBase := Src.GetRawPos(sx, sy, 0);
        Move(Src.FData[SrcBase], V.FData[VBase], Dep * csNeuralFloatSize);
      end
      else
        FillChar(V.FData[VBase], Dep * csNeuralFloatSize, 0); // neutral gray fill (pixel 128)
      Inc(VBase, Dep);
    end;
  end;
  Src.Free;
end;

// --- Photometric ops -------------------------------------------------------

procedure AugAutoContrast(V: TNNetVolume);
var
  d, x, y: integer;
  lo, hi, p, scale: TNeuralFloat;
  DepthM1, SizeYM1, SizeXM1, Pos, XStride: integer;
begin
  // Per-channel min/max stretch to full 0..255 range (torchvision autocontrast).
  DepthM1 := V.Depth - 1;
  SizeYM1 := V.SizeY - 1;
  SizeXM1 := V.SizeX - 1;
  XStride := V.GetRawPos(1, 0, 0);
  for d := 0 to DepthM1 do
  begin
    lo := 255; hi := 0;
    for y := 0 to SizeYM1 do
    begin
      Pos := V.GetRawPos(0, y, d);
      for x := 0 to SizeXM1 do
      begin
        p := AugClampPixel(AugNeuronToPixel(V.FData[Pos]));
        if p < lo then lo := p;
        if p > hi then hi := p;
        Inc(Pos, XStride);
      end;
    end;
    if hi <= lo then continue;
    scale := 255.0 / (hi - lo);
    for y := 0 to SizeYM1 do
    begin
      Pos := V.GetRawPos(0, y, d);
      for x := 0 to SizeXM1 do
      begin
        p := AugClampPixel(AugNeuronToPixel(V.FData[Pos]));
        p := (p - lo) * scale;
        V.FData[Pos] := AugClampNeuron(AugPixelToNeuron(AugClampPixel(p)));
        Inc(Pos, XStride);
      end;
    end;
  end;
end;

procedure AugEqualize(V: TNNetVolume);
var
  d, x, y, i, b, total: integer;
  hist: array[0..255] of integer;
  cdf: array[0..255] of integer;
  lut: array[0..255] of TNeuralFloat;
  cdfMin, denom, p: TNeuralFloat;
  acc: integer;
  DepthM1, SizeYM1, SizeXM1, Pos, XStride: integer;
begin
  // Per-channel histogram equalization (torchvision equalize).
  DepthM1 := V.Depth - 1;
  SizeYM1 := V.SizeY - 1;
  SizeXM1 := V.SizeX - 1;
  XStride := V.GetRawPos(1, 0, 0);
  for d := 0 to DepthM1 do
  begin
    FillChar(hist, SizeOf(hist), 0);
    for y := 0 to SizeYM1 do
    begin
      Pos := V.GetRawPos(0, y, d);
      for x := 0 to SizeXM1 do
      begin
        b := Round(AugClampPixel(AugNeuronToPixel(V.FData[Pos])));
        if b < 0 then b := 0; if b > 255 then b := 255;
        Inc(hist[b]);
        Inc(Pos, XStride);
      end;
    end;
    acc := 0;
    cdfMin := -1;
    for i := 0 to 255 do
    begin
      Inc(acc, hist[i]);
      cdf[i] := acc;
      if (cdfMin < 0) and (hist[i] > 0) then cdfMin := acc;
    end;
    total := V.SizeX * V.SizeY;
    denom := total - cdfMin;
    if denom <= 0 then
    begin
      // Flat channel: identity LUT.
      for i := 0 to 255 do lut[i] := i;
    end
    else
      for i := 0 to 255 do
        lut[i] := AugClampPixel(((cdf[i] - cdfMin) / denom) * 255.0);
    for y := 0 to SizeYM1 do
    begin
      Pos := V.GetRawPos(0, y, d);
      for x := 0 to SizeXM1 do
      begin
        b := Round(AugClampPixel(AugNeuronToPixel(V.FData[Pos])));
        if b < 0 then b := 0; if b > 255 then b := 255;
        p := lut[b];
        V.FData[Pos] := AugClampNeuron(AugPixelToNeuron(p));
        Inc(Pos, XStride);
      end;
    end;
  end;
end;

function AugChannelMeanGray(V: TNNetVolume): TNeuralFloat;
var
  I: integer;
  s: TNeuralFloat;
  SizeM1: integer;
begin
  // Luminance-ish mean over all pixels (used by Color/Contrast degenerate).
  s := 0;
  SizeM1 := V.Size - 1;
  for I := 0 to SizeM1 do
    s := s + AugClampPixel(AugNeuronToPixel(V.FData[I]));
  if V.Size > 0 then Result := s / V.Size else Result := 128;
end;

procedure AugColor(V: TNNetVolume; Factor: TNeuralFloat);
var
  W, H, Dep, x, y, d: integer;
  gray, p, OneMinusFactor, pixelAdd: TNeuralFloat;
  WM1, HM1, DepM1, Base: integer;
begin
  // Saturation adjustment: blend each pixel toward its per-pixel grayscale.
  W := V.SizeX; H := V.SizeY; Dep := V.Depth;
  if Dep < 2 then
  begin
    // Single channel: color has no effect.
    Exit;
  end;
  WM1 := W - 1; HM1 := H - 1; DepM1 := Dep - 1;
  OneMinusFactor := 1.0 - Factor;
  for y := 0 to HM1 do
    for x := 0 to WM1 do
    begin
      Base := V.GetRawPos(x, y, 0);
      gray := 0;
      for d := 0 to DepM1 do
        gray := gray + AugClampPixel(AugNeuronToPixel(V.FData[Base + d]));
      gray := gray / Dep;
      pixelAdd := OneMinusFactor * gray;
      for d := 0 to DepM1 do
      begin
        p := AugClampPixel(AugNeuronToPixel(V.FData[Base + d]));
        p := Factor * p + pixelAdd;
        V.FData[Base + d] := AugClampNeuron(AugPixelToNeuron(AugClampPixel(p)));
      end;
    end;
end;

procedure AugContrast(V: TNNetVolume; Factor: TNeuralFloat);
begin
  // Blend toward the global mean gray.
  AugBlendTowardScalar(V, AugChannelMeanGray(V), Factor);
end;

procedure AugBrightness(V: TNNetVolume; Factor: TNeuralFloat);
begin
  // Blend toward black (pixel 0).
  AugBlendTowardScalar(V, 0.0, Factor);
end;

procedure AugSharpness(V: TNNetVolume; Factor: TNeuralFloat);
var
  W, H, Dep, x, y, d, ix, iy: integer;
  Src: TNNetVolume;
  acc, wsum, p, smooth: TNeuralFloat;
  kw, OneMinusFactor: TNeuralFloat;
  WM1, HM1, DepM1, CtrPos: integer;
begin
  // Blend toward a 3x3 box-blurred image (torchvision uses a smoothing kernel;
  // a box blur is a close, dependency-free stand-in). Factor>1 sharpens.
  W := V.SizeX; H := V.SizeY; Dep := V.Depth;
  if (W < 3) or (H < 3) then Exit;
  Src := TNNetVolume.Create;
  Src.Copy(V);
  WM1 := W - 1; HM1 := H - 1; DepM1 := Dep - 1;
  OneMinusFactor := 1.0 - Factor;
  for d := 0 to DepM1 do
    for y := 0 to HM1 do
      for x := 0 to WM1 do
      begin
        // Interior pixels get blurred; the 1px border is left unchanged
        // (matches torchvision which keeps the border).
        if (x = 0) or (y = 0) or (x = W - 1) or (y = H - 1) then continue;
        acc := 0; wsum := 0;
        for iy := -1 to 1 do
          for ix := -1 to 1 do
          begin
            if (ix = 0) and (iy = 0) then kw := 5 else kw := 1;
            acc := acc + kw * AugClampPixel(AugNeuronToPixel(Src[x + ix, y + iy, d]));
            wsum := wsum + kw;
          end;
        smooth := acc / wsum;
        CtrPos := Src.GetRawPos(x, y, d);
        p := AugClampPixel(AugNeuronToPixel(Src.FData[CtrPos]));
        p := Factor * p + OneMinusFactor * smooth;
        V.FData[CtrPos] := AugClampNeuron(AugPixelToNeuron(AugClampPixel(p)));
      end;
  Src.Free;
end;

procedure AugPosterize(V: TNNetVolume; Bits: integer);
var
  mask, I, b: integer;
  SizeM1: integer;
begin
  if Bits < 1 then Bits := 1;
  if Bits > 8 then Bits := 8;
  if Bits = 8 then Exit; // identity
  mask := (255 shl (8 - Bits)) and 255;
  SizeM1 := V.Size - 1;
  for I := 0 to SizeM1 do
  begin
    b := Round(AugClampPixel(AugNeuronToPixel(V.FData[I])));
    if b < 0 then b := 0; if b > 255 then b := 255;
    b := b and mask;
    V.FData[I] := AugClampNeuron(AugPixelToNeuron(b));
  end;
end;

procedure AugSolarize(V: TNNetVolume; Threshold: TNeuralFloat);
var
  I: integer;
  p: TNeuralFloat;
  SizeM1: integer;
begin
  SizeM1 := V.Size - 1;
  for I := 0 to SizeM1 do
  begin
    p := AugClampPixel(AugNeuronToPixel(V.FData[I]));
    if p >= Threshold then p := 255 - p;
    V.FData[I] := AugClampNeuron(AugPixelToNeuron(AugClampPixel(p)));
  end;
end;

// --- Geometric ops ---------------------------------------------------------

procedure AugRotate(V: TNNetVolume; DegAngle: TNeuralFloat);
var
  rad, c, s: TNeuralFloat;
  Mat: array[0..5] of TNeuralFloat;
begin
  if DegAngle = 0 then Exit; // bit-identity
  rad := DegAngle * Pi / 180.0;
  pcr_sincosf(rad, s, c);
  // Inverse rotation (dst -> src) so we sample. Rotating image by +theta uses
  // src = R(-theta)*dst.
  Mat[0] := c;  Mat[1] := s;  Mat[2] := 0;
  Mat[3] := -s; Mat[4] := c;  Mat[5] := 0;
  AugAffineWarp(V, Mat);
end;

procedure AugShearX(V: TNNetVolume; ShearFactor: TNeuralFloat);
var
  Mat: array[0..5] of TNeuralFloat;
begin
  if ShearFactor = 0 then Exit; // bit-identity
  // src_x = dst_x - shear*dst_y (inverse of x' = x + shear*y).
  Mat[0] := 1; Mat[1] := -ShearFactor; Mat[2] := 0;
  Mat[3] := 0; Mat[4] := 1;            Mat[5] := 0;
  AugAffineWarp(V, Mat);
end;

procedure AugShearY(V: TNNetVolume; ShearFactor: TNeuralFloat);
var
  Mat: array[0..5] of TNeuralFloat;
begin
  if ShearFactor = 0 then Exit; // bit-identity
  Mat[0] := 1;            Mat[1] := 0; Mat[2] := 0;
  Mat[3] := -ShearFactor; Mat[4] := 1; Mat[5] := 0;
  AugAffineWarp(V, Mat);
end;

procedure AugTranslateX(V: TNNetVolume; Pixels: TNeuralFloat);
var
  Mat: array[0..5] of TNeuralFloat;
begin
  if Pixels = 0 then Exit; // bit-identity
  // Shift content by +Pixels -> sample from src_x = dst_x - Pixels.
  Mat[0] := 1; Mat[1] := 0; Mat[2] := -Pixels;
  Mat[3] := 0; Mat[4] := 1; Mat[5] := 0;
  AugAffineWarp(V, Mat);
end;

procedure AugTranslateY(V: TNNetVolume; Pixels: TNeuralFloat);
var
  Mat: array[0..5] of TNeuralFloat;
begin
  if Pixels = 0 then Exit; // bit-identity
  Mat[0] := 1; Mat[1] := 0; Mat[2] := 0;
  Mat[3] := 0; Mat[4] := 1; Mat[5] := -Pixels;
  AugAffineWarp(V, Mat);
end;

// --- Magnitude mapping (torchvision _AUGMENTATION_SPACE) --------------------

function AugMagFrac(Magnitude: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  // Maps M in 0..NeuralAugMaxMagnitude to t in 0..1.
  if NeuralAugMaxMagnitude <= 0 then Result := 0
  else Result := Magnitude / NeuralAugMaxMagnitude;
  if Result < 0 then Result := 0;
  if Result > 1 then Result := 1;
end;

function AugRandSign: TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
begin
  if Random(2) = 0 then Result := 1.0 else Result := -1.0;
end;

procedure NeuralAugApplyOp(V: TNNetVolume; Op: TNeuralAugOp; Magnitude: integer);
var
  t, sgn: TNeuralFloat;
  W, H: integer;
begin
  if (V = nil) or (V.Size = 0) then Exit;
  W := V.SizeX; H := V.SizeY;
  t := AugMagFrac(Magnitude);
  case Op of
    csaIdentity: ; // no-op
    csaAutoContrast: AugAutoContrast(V);   // parameter-free
    csaEqualize:     AugEqualize(V);       // parameter-free
    csaRotate:
      begin
        sgn := AugRandSign;
        AugRotate(V, sgn * t * 30.0);      // up to +/-30 deg
      end;
    csaShearX:
      begin
        sgn := AugRandSign;
        AugShearX(V, sgn * t * 0.3);       // up to +/-0.3
      end;
    csaShearY:
      begin
        sgn := AugRandSign;
        AugShearY(V, sgn * t * 0.3);
      end;
    csaTranslateX:
      begin
        sgn := AugRandSign;
        AugTranslateX(V, sgn * t * (W * 0.45));
      end;
    csaTranslateY:
      begin
        sgn := AugRandSign;
        AugTranslateY(V, sgn * t * (H * 0.45));
      end;
    csaPosterize:
      // torchvision: bits = 8 - round(t*4); M=0 -> 8 bits (identity).
      AugPosterize(V, 8 - Round(t * 4));
    csaSolarize:
      // threshold = 255*(1-t); M=0 -> 255 (identity).
      AugSolarize(V, 255.0 * (1.0 - t));
    csaColor:
      AugColor(V, 1.0 + AugRandSign * t * 0.9);
    csaContrast:
      AugContrast(V, 1.0 + AugRandSign * t * 0.9);
    csaBrightness:
      AugBrightness(V, 1.0 + AugRandSign * t * 0.9);
    csaSharpness:
      AugSharpness(V, 1.0 + AugRandSign * t * 0.9);
  end;
end;

procedure NeuralRandAugment(V: TNNetVolume; N: integer; Magnitude: integer);
var
  i, opIdx, opCount: integer;
begin
  if (V = nil) or (V.Size = 0) or (N <= 0) then Exit;
  opCount := Ord(High(TNeuralAugOp)) + 1;
  for i := 1 to N do
  begin
    // Uniform over the WHOLE bank including identity (torchvision includes it).
    opIdx := Random(opCount);
    NeuralAugApplyOp(V, TNeuralAugOp(opIdx), Magnitude);
  end;
end;

procedure NeuralTrivialAugment(V: TNNetVolume);
var
  opIdx, opCount, mag: integer;
begin
  if (V = nil) or (V.Size = 0) then Exit;
  opCount := Ord(High(TNeuralAugOp)) + 1;
  opIdx := Random(opCount);
  mag := Random(NeuralAugMaxMagnitude + 1); // uniform 0..max inclusive
  NeuralAugApplyOp(V, TNeuralAugOp(opIdx), mag);
end;

procedure NeuralRandomErasing(V: TNNetVolume;
  pProb: TNeuralFloat; pAreaLow, pAreaHigh, pAspectLow, pFill: TNeuralFloat);
var
  W, H, Dep, area, x0, y0, ew, eh, x, y, d, attempt: integer;
  targetArea, aspect, logLo, logHi: TNeuralFloat;
  YMax, XMax, DepM1, Base, RunLen: integer;
begin
  if (V = nil) or (V.Size = 0) then Exit;
  if Random >= pProb then Exit;
  W := V.SizeX; H := V.SizeY; Dep := V.Depth;
  if (W <= 0) or (H <= 0) then Exit;
  area := W * H;
  logLo := pcr_logf(pAspectLow);
  logHi := pcr_logf(1.0 / pAspectLow);
  for attempt := 1 to 10 do
  begin
    targetArea := (pAreaLow + Random * (pAreaHigh - pAreaLow)) * area;
    aspect := Exp(logLo + Random * (logHi - logLo));
    ew := Round(Sqrt(targetArea / aspect));
    eh := Round(Sqrt(targetArea * aspect));
    if (ew > 0) and (ew < W) and (eh > 0) and (eh < H) then
    begin
      x0 := Random(W - ew + 1);
      y0 := Random(H - eh + 1);
      YMax := y0 + eh - 1;
      XMax := x0 + ew - 1;
      DepM1 := Dep - 1;
      RunLen := ew * Dep; // whole x-run per row is contiguous
      for y := y0 to YMax do
      begin
        Base := V.GetRawPos(x0, y, 0);
        if pFill = 0 then
          FillChar(V.FData[Base], RunLen * csNeuralFloatSize, 0)
        else
          for d := 0 to RunLen - 1 do
            V.FData[Base + d] := pFill;
      end;
      Exit;
    end;
  end;
end;

{ TNeuralAugmentationPolicy }

constructor TNeuralAugmentationPolicy.Create(pPolicy: TNeuralAugPolicy;
  pNumOps: integer; pMagnitude: integer; pErasingProb: TNeuralFloat);
begin
  inherited Create;
  FPolicy := pPolicy;
  FNumOps := pNumOps;
  FMagnitude := pMagnitude;
  FErasingProb := pErasingProb;
end;

procedure TNeuralAugmentationPolicy.Augment(pInput: TNNetVolume; ThreadId: integer);
begin
  case FPolicy of
    napRandAugment:    NeuralRandAugment(pInput, FNumOps, FMagnitude);
    napTrivialAugment: NeuralTrivialAugment(pInput);
  end;
  if FErasingProb > 0 then
    NeuralRandomErasing(pInput, FErasingProb);
end;

end.
