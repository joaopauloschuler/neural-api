(*
neuraltokenizer
Copyright (C) 2024 Joao Paulo Schwarz Schuler

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
unit neuraltokenizer;
{$include neuralnetwork.inc}

interface

uses
Classes, SysUtils, neuralvolume;

type
  TPairFrequency = TStringListInt;
  TMergeList = TNNetStringList;

  { TNeuralTokenizer }
  TNeuralTokenizer = class(TStringListInt)
    public
      function GetVocabCount(): integer; override;
      procedure Tokenize(const pText: string; Result: TIntegerList); overload;
      procedure Tokenize(pString: string; var IntArr: TNeuralIntegerArray); overload; override;
      procedure TokenizeWord(const pText: string; Result: TIntegerList);
      function DeTokenize(TokenId: integer): string; overload; override;
      function DeTokenize(TokenIds: TIntegerList): string; overload;
      function TokenizerHasSeparator: boolean; override;

      procedure FitOnFile(const FileName: string; VocabSize: integer;
        WriteDebug: boolean = true);
      procedure TokenizeFileToCsv(const InputFileName, OutputFileName: string);
      // This function will be removed - do not use it.
      procedure Test;
  end;

procedure TestNeuralTokenizer();

implementation

uses Math;

type
  TTokenizerBase = class
    public
      class procedure CountPairs(const vocab: TNNetStringList; var pairCounts: TPairFrequency); static;
      class function GetMostFrequentPair(const pairCounts: TPairFrequency): string; static;
      class procedure MergePair(var vocab: TNNetStringList; const pair: string; const replacement: string); static;
      class procedure Encode(var vocab: TNNetStringList; numMerges: integer;
        var merges: TMergeList; MergesPerLoop: integer = 1;
        WriteDebug: boolean = true); static;
      class function LoadDataset(const filename: string): TNNetStringList; static;
      class function InsertCharBetweenChars(const input: string; const SpaceChar:char = ' '): string; static;
  end;

const csBPESeparator:char = chr(30);

// Function to count pair frequencies in the vocabulary
class procedure TTokenizerBase.CountPairs(const vocab: TNNetStringList; var pairCounts: TPairFrequency);
var
  i, j: integer;
  pair: string;
  PairIndex: integer;
  MaxVocab, MaxLen: integer;
  Tokens: TNNetStringList;
  CurrentToken, NexToken: string;
  ShouldQuit: boolean;
begin
  pairCounts.Clear;
  pairCounts.Sorted := true;
  MaxVocab := vocab.Count - 1;
  Tokens := CreateTokenizedStringList(csBPESeparator);
  ShouldQuit := false;
  for i := 0 to MaxVocab do
  begin
    Tokens.DelimitedText := vocab[i];
    MaxLen := Tokens.Count - 2;
    if MaxLen > -1 then
    begin
      for j := 0 to MaxLen do
      begin
        CurrentToken := Tokens[j];
        NexToken := Tokens[j+1];
        if
          (((CurrentToken >= 'a') and (CurrentToken < '{')) or (CurrentToken='''')) and
          (((NexToken >= 'a') and (NexToken < '{')) or (NexToken='''')) then
        begin
          pair := CurrentToken + csBPESeparator + NexToken;
          PairIndex := pairCounts.IndexOf(pair);
          if PairIndex = -1
          then pairCounts.AddInteger(pair, 1)
          else
          begin
            pairCounts.Integers[PairIndex] := pairCounts.Integers[PairIndex] + 1;
            if pairCounts.Integers[PairIndex] > 32 then ShouldQuit := true;
          end;
        end;
        if ShouldQuit then break;
      end;
    end;
    if ShouldQuit then break;
  end;
  Tokens.Free;
end;

// Function to find the most frequent pair
class function TTokenizerBase.GetMostFrequentPair(const pairCounts: TPairFrequency): string;
var
  i, maxCount: integer;
  maxPair: string;
begin
  maxCount := 0;
  maxPair := '';
  for i := 0 to pairCounts.Count - 1 do
  begin
    if pairCounts.Integers[i] > maxCount then
    begin
      maxCount := pairCounts.Integers[i];
      maxPair := pairCounts[i];
    end;
  end;
  //Write('[',maxCount,']');
  Result := maxPair;
end;

// Function to merge a pair in the vocabulary
class procedure TTokenizerBase.MergePair(var vocab: TNNetStringList; const pair: string; const replacement: string);
var
  i: integer;
begin
  for i := 0 to vocab.Count - 1 do
    vocab[i] := StringReplace(vocab[i], pair, replacement, [rfReplaceAll]);
end;

// Main BPE algorithm
class procedure TTokenizerBase.Encode(var vocab: TNNetStringList;
  numMerges: integer;
  var merges: TMergeList;
  MergesPerLoop: integer = 1;
  WriteDebug: boolean = true);
var
  pairCounts: TPairFrequency;
  I, J, MaxJ: integer;
  pair, newToken: string;
  StartTime: double;
  MergesLoopCnt: integer;
begin
  pairCounts := TPairFrequency.Create;
  MergesLoopCnt := (numMerges-128) div MergesPerLoop;
  try
    for I := 0 to MergesLoopCnt - 1 do
    begin
      StartTime := Now();
      if WriteDebug then Write(I*MergesPerLoop, ' Counting pairs. ');
      CountPairs(vocab, pairCounts);
      if WriteDebug then Write(FloatToStrF((Now()-StartTime) * 24 * 60 * 60 ,ffFixed,6,2) + 's. ');
      //pair := GetMostFrequentPair(pairCounts);
      if pairCounts.Count>1 then
      begin
        pairCounts.SortByIntegerDesc;
        MaxJ := Min(MergesPerLoop - 1, pairCounts.Count);
        for J := 0 to MaxJ do
        begin
          pair := pairCounts[J];
          if pair = '' then
            break;
          newToken := StringReplace(pair, csBPESeparator, '', [rfReplaceAll]);
          MergePair(vocab, pair, newToken);
          merges.Values[pair] := newToken;
          if WriteDebug then WriteLn(
            ' : ', pair, ' -> ', newToken,
            ' [',pairCounts.Integers[J],']',
            ' Time: ', FloatToStrF((Now()-StartTime) * 24 * 60 * 60 ,ffFixed,6,2) + 's');
        end;
      end;
    end;
  finally
    pairCounts.Free;
  end;
end;

// Function to load vocabulary from file
class function TTokenizerBase.LoadDataset(const filename: string): TNNetStringList;
var
  lines: TStringList;
  i: integer;
begin
  Result := TNNetStringList.Create;
  lines := CreateQuotedTokenizedStringList(' ', chr(29));
  try
    lines.LoadFromFile(filename);
    lines.DelimitedText := lines.Text;
    for i := 0 to lines.Count - 1 do
    begin
      //WriteLn(InsertSpacesBetweenChars(lines[i],csBPESeparator));
      Result.Add(InsertCharBetweenChars(Trim(lines[i]),csBPESeparator));
    end;
  finally
    lines.Free;
  end;
end;

class function TTokenizerBase.InsertCharBetweenChars(const input: string; const SpaceChar:char = ' '): string;
var
  i: Integer;
  len: Integer;
begin
  len := Length(input);
  if len <= 1 then
    Exit(input);  // Return original string if it's empty or has only one character

  SetLength(Result, len * 2 - 1);  // Pre-allocate space for efficiency

  // Insert characters and spaces
  for i := 1 to len do
  begin
    Result[i * 2 - 1] := input[i];
    if i < len then
      Result[i * 2] := SpaceChar;
  end;
end;

function TNeuralTokenizer.GetVocabCount(): integer;
begin
  Result := Count + 128;
end;

{ TNeuralTokenizer }
procedure TNeuralTokenizer.Tokenize(const pText: string;
  Result: TIntegerList);
var
  Tokens, Words: TNNetStringList;
  i: integer;
  current, KeyName: string;
  TokenLen: integer;
  TokenId: integer;
  MaxWord, CntWord: integer;
begin
  Words := CreateQuotedTokenizedStringList(pText, ' ', chr(29));
  Tokens := CreateQuotedTokenizedStringList(csBPESeparator, chr(29));
  MaxWord := Words.Count - 1;
  for CntWord := 0 to MaxWord do
  begin
    if CntWord > 0 then
    begin
      Result.Add(Ord(' '));
      //WriteLn( 'space' );
    end;
    current := Words[CntWord];
    TokenLen := length(current);
    if TokenLen=1 then
    begin
      //WriteLn(current, ' -> Fast ORD: ',Ord(current[1]));
      {$IFDEF DEBUG}
      if Ord(current[1]) > 127 then WriteLn('ERROR - TNeuralTokenizer.Tokenize [A]:', Ord(current[1]));
      {$ENDIF}
      Result.Add(Ord(current[1]));
    end
    else if TokenLen>1 then
    begin
      KeyName := current;
      TokenId := Self.IndexOf(KeyName);
      if TokenId >= 0 then
      begin
        //WriteLn(current, ' -> Fast ID: ', TokenId+128);
        {$IFDEF DEBUG}
        if TokenId > Self.Count then WriteLn('ERROR - TNeuralTokenizer.Tokenize [B]:', TokenId);
        {$ENDIF}
        Result.Add(TokenId+128);
      end
      else
      begin
        TokenizeWord(current, Result);
      end;
    end;
  end; // for CntWord

  Words.Free;
  Tokens.Free;
end;

procedure TNeuralTokenizer.Tokenize(pString: string;
  var IntArr: TNeuralIntegerArray);
var
  IL: TIntegerList;
  MaxIL: integer;
  ILCount: integer;
begin
  IL := TIntegerList.Create();
  Tokenize(pString, IL);
  SetLength(IntArr, IL.Count);
  MaxIL := IL.Count - 1;
  if MaxIL > -1 then
  begin
    for ILCount := 0 to MaxIL do
    begin
      IntArr[ILCount] := IL[ILCount];
    end;
  end;
  IL.Free;
end;

procedure TNeuralTokenizer.TokenizeWord(const pText: string; Result: TIntegerList);
var
  LocalText: string;
  FirstChar: char;
  CurrentTokenId, LocalTokenId, CurrentPos, LocalLen: integer;
begin
  LocalText := pText;
  LocalLen := Length(LocalText);
  while LocalLen > 0 do
  begin
    FirstChar := LocalText[1];
    CurrentTokenId := Ord(FirstChar);
    if LocalLen = 1 then
    begin
      {$IFDEF DEBUG}
      if CurrentTokenId > 127 then WriteLn('ERROR TNeuralTokenizer.TokenizeWord [C]:', CurrentTokenId);
      {$ENDIF}
      Result.Add(CurrentTokenId);
      LocalLen := 0;
      LocalText := '';
      //WriteLn(FirstChar, ' -> ',CurrentTokenId);
    end
    else if not(((FirstChar >= 'a') and (FirstChar < '{')) or (FirstChar='''')) then
    begin
      {$IFDEF DEBUG}
      if CurrentTokenId > 127 then WriteLn('ERROR TNeuralTokenizer.TokenizeWord [D]:', CurrentTokenId);
      {$ENDIF}
      Result.Add(CurrentTokenId);
      LocalLen := LocalLen - 1;
      LocalText := copy(LocalText, 2, LocalLen);
      //WriteLn(FirstChar, ' -> ',CurrentTokenId);
    end
    else
    begin
      LocalTokenId := CurrentTokenId;
      CurrentPos := 1;
      while (LocalTokenId > -1) and (CurrentPos <= LocalLen) do
      begin
        CurrentTokenId := LocalTokenId;
        Inc(CurrentPos);
        LocalTokenId := Self.WordToInteger(copy(LocalText, 1, CurrentPos));
      end;
      if (CurrentPos > 2) then
      begin
        Dec(CurrentPos);
        //WriteLn(copy(LocalText, 1, CurrentPos), ' -> ',CurrentTokenId+128);
        {$IFDEF DEBUG}
        if CurrentTokenId > Self.Count then WriteLn('ERROR TNeuralTokenizer.TokenizeWord [E]:', CurrentTokenId);
        {$ENDIF}
        Result.Add(CurrentTokenId+128);
        LocalLen := LocalLen - CurrentPos;
        LocalText := copy(LocalText, CurrentPos+1, LocalLen);
      end
      else
      begin
        {$IFDEF DEBUG}
        if CurrentTokenId > 127 then WriteLn('ERROR TNeuralTokenizer.TokenizeWord [F]:', CurrentTokenId);
        {$ENDIF}
        Result.Add(CurrentTokenId);
        LocalLen := LocalLen - 1;
        LocalText := copy(LocalText, 2, LocalLen);
        //WriteLn(FirstChar, ' -> ',CurrentTokenId);
      end;
    end;
  end; // of while
end;

function TNeuralTokenizer.DeTokenize(TokenId: integer): string;
begin
  if TokenId < 128 then
  begin
    Result := Chr(TokenId);
  end
  else
  begin
    Result := IntegerToWord(TokenId-128);
  end;
end;

// TODO: code a faster implementation.
function TNeuralTokenizer.DeTokenize(TokenIds: TIntegerList): string;
var
  TokenCnt, MaxToken: integer;
begin
  Result := '';
  MaxToken := TokenIds.Count - 1;
  if MaxToken >= 0 then
  begin
    for TokenCnt := 0 to MaxToken do
    begin
      Result := Result + DeTokenize(TokenIds[TokenCnt]);
    end;
  end;
end;

function TNeuralTokenizer.TokenizerHasSeparator: boolean;
begin
  Result := false;
end;

procedure TNeuralTokenizer.FitOnFile(const FileName: string; VocabSize: integer;
  WriteDebug: boolean = true);
var
  vocab: TNNetStringList;
  merges: TMergeList;
  i: integer;
  Tokens: TIntegerList;
begin
  try
    // Load vocabulary from file
    vocab := TTokenizerBase.LoadDataset(FileName);
    Tokens := TIntegerList.Create;

    // Perform Encoding
    merges := TMergeList.Create;
    merges.Sorted := true;

    TTokenizerBase.Encode(vocab, VocabSize, merges, 1, WriteDebug);

    Self.Clear;
    Self.Sorted := true;
    for i := 0 to merges.Count - 1 do
    begin
      Self.Add(merges.ValueFromIndex[i]);
    end;
    SaveCurrentPosition();
  finally
    Tokens.Free;
    vocab.Free;
    merges.Free;
  end;
end;

procedure TNeuralTokenizer.TokenizeFileToCsv(const InputFileName,
  OutputFileName: string);
var
  LargeFileIn, LargeFileOut: TextFile;
  StrLine: string;
  Tokens: TIntegerList;
begin
  Tokens := TIntegerList.Create;
  AssignFile(LargeFileIn, InputFileName);
  AssignFile(LargeFileOut, OutputFileName);
  Reset(LargeFileIn);
  ReWrite(LargeFileOut);
  while not Eof(LargeFileIn) do
  begin
    ReadLn(LargeFileIn, StrLine);
    Tokens.Clear;
    Tokenize(StrLine, Tokens);
    WriteLn(LargeFileOut, IntegerListToCsv(Tokens));
    {$IFDEF Debug}
    if StrLine <> DeTokenize(Tokens) then
    begin
      WriteLn('Error:');
      WriteLn('Input: ',StrLine);
      WriteLn('Output:',DeTokenize(Tokens));
    end;
    {$ENDIF}
  end;
  CloseFile(LargeFileIn);
  CloseFile(LargeFileOut);
  Tokens.Free;
end;

procedure TNeuralTokenizer.Test;
var
  vocab: TNNetStringList;
  merges: TMergeList;
  i: integer;
  Tokens: TIntegerList;
begin
  try
    // Load vocabulary from file
    vocab := TTokenizerBase.LoadDataset('datasets/tinystories-10k.txt');
    //vocab := TTokenizerBase.LoadDataset('datasets/small_ds.txt');
    Tokens := TIntegerList.Create;
    WriteLn('Loaded dataset.');
    //for i := 0 to vocab.Count - 1 do
    //  WriteLn(vocab[i]);
    //WriteLn;

    // Perform Encoding
    merges := TMergeList.Create;
    merges.Sorted := true;
    WriteLn('Encoding vocabulary.');
//    TTokenizerBase.Encode(vocab, 1024*3, merges, 128);
    TTokenizerBase.Encode(vocab, 3000, merges, 1);

    // Save merges to file
    // merges.SaveToFile('merges.txt');
    // WriteLn('Merges saved to merges.txt');

    WriteLn;
    WriteLn('Final vocabulary has:',merges.Count - 1,' elements.');
    Self.Clear;
    Self.Sorted := true;
    for i := 0 to merges.Count - 1 do
    begin
      Self.Add(merges.ValueFromIndex[i]);
    end;
    SaveCurrentPosition();

    Self.SaveToFile('test-dict.txt');

    // Tokenize new text
    WriteLn;
    WriteLn('Tokenizing new text:');
    Tokenize('lowest 12 3 . lower', Tokens);
    WriteLn('Tokens: ');
    for i := 0 to Tokens.Count - 1 do
      WriteLn(i,' : ',Tokens[i]);
    WriteLn(DeTokenize(Tokens));
    ReadLn;
  finally
    Tokens.Free;
    vocab.Free;
    merges.Free;
  end;
end;

procedure TestNeuralTokenizer();
var
  X: TNeuralTokenizer;
begin
  X := TNeuralTokenizer.Create;
  X.FitOnFile('datasets/tinystories-10k.txt', 3000, true);
  X.SaveToFile('datasets/experiment-tinystories-vocab-3k-cai.txt');
  //X.LoadVocabularyFromFile('datasets/tinystories-vocab-3k-cai.txt');
  //X.TokenizeFileToCsv('datasets/tinystories.txt','datasets/tinystories-2.1M-tokenized3k.csv');
  X.Free;
end;

end.
