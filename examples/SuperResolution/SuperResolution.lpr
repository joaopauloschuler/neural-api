program SuperResolution;
(*
 Coded by Joao Paulo Schwarz Schuler.
 // https://sourceforge.net/p/cai
 This command line tool runs the CAI Convolutional Neural Network with
 CIFAR10 files.

 In the case that your processor supports AVX instructions, uncomment
 {$DEFINE AVX} or {$DEFINE AVX2} defines. Also have a look at AVX512 define.

 Look at TTestCNNAlgo.WriteHelp; for more info.
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  CustApp,
  neuralnetwork,
  neuralvolume,
  Math,
  neuraldatasets,
  neuralfit,
  FPImage,
  usuperresolutionexample,
  FPWriteBMP, FPWritePCX, FPWriteJPEG, FPWritePNG,
  FPWritePNM, FPWriteTGA, FPWriteTiff;

type
  { TSuperResolutionExample }
  TSuperResolutionExample = class(TCustomApplication)
  protected
    procedure DoRun; override;
    procedure RunAlgo(inputFile, outputFile: string);
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
    procedure WriteHelp; virtual;
  end;

  { TSuperResolutionExample }
  procedure TSuperResolutionExample.DoRun;
  var
    inputFile, outputFile: string;
  begin
    // quick check parameters
    // parse parameters
    if HasOption('h', 'help') then
    begin
      WriteHelp;
    end;

    if HasOption('i', 'input') then
    begin
      inputFile := GetOptionValue('i', 'input');
    end
    else
    begin
      WriteHelp;
      Terminate;
      Exit;
    end;

    outputFile := 'superresolution.png';
    if HasOption('o', 'output') then
    begin
      outputFile := GetOptionValue('o', 'output');
    end;

    if FileExists(inputFile) then
    begin
      RunAlgo(inputFile, outputFile);
    end
    else
    begin
      WriteLn('File can''t be found: ', inputFile);
    end;

    Terminate;
    Exit;
  end;

  const
    csTileSize = 16;
    csTileBorder = 4;

  procedure TSuperResolutionExample.RunAlgo(inputFile, outputFile: string);
  var
    Image: TFPMemoryImage;
    NN: TNNet;
    InputImgVol, AuxInputImgVol, OutputImgVol: TNNetVolume;
    InputTile, OutputTile: TNNetVolume;
    MaxTileX, MaxTileY: integer;
    TileXCnt, TileYCnt: integer;
    TileStartX, TileStartY: integer;
  begin
    InputImgVol := TNNetVolume.Create();
    OutputImgVol := TNNetVolume.Create();
    Image := TFPMemoryImage.Create(1,1);
    WriteLn('Loading input file: ', inputFile);
    Image.LoadFromFile(inputFile);
    LoadImageIntoVolume(Image, InputImgVol);
    WriteLn('Input image size: ', InputImgVol.SizeX,' x ', InputImgVol.SizeY,' x ',InputImgVol.Depth);
    InputImgVol.Divi(64);
    InputImgVol.Sub(2);
    WriteLn('Creating Neural Network...');
    if InputImgVol.SizeX * InputImgVol.SizeY <= 128*128 then
    begin
      WriteLn('Resizing...');
      NN := CreateResizingNN(InputImgVol.SizeX, InputImgVol.SizeY, csExampleFileName);
      NN.Compute(InputImgVol);
      NN.GetOutput(OutputImgVol);
    end
    else
    begin
      WriteLn('Resizing with tiles...');
      NN := CreateResizingNN(csTileSize+csTileBorder*2, csTileSize+csTileBorder*2, csExampleFileName);
      InputTile := TNNetVolume.Create(csTileSize+csTileBorder*2, csTileSize+csTileBorder*2, 3);
      OutputTile := TNNetVolume.Create(csTileSize * 2 + csTileBorder*2, csTileSize * 2 + csTileBorder*2, 3);
      MaxTileX := (InputImgVol.SizeX div csTileSize) - 1;
      MaxTileY := (InputImgVol.SizeY div csTileSize) - 1;
      if
        (InputImgVol.SizeX mod csTileSize < csTileBorder*2) or
        (InputImgVol.SizeY mod csTileSize < csTileBorder*2) then
      begin
        WriteLn('Padding input image.');
        AuxInputImgVol := TNNetVolume.Create();
        AuxInputImgVol.CopyPadding(InputImgVol, csTileBorder);
        InputImgVol.Copy(AuxInputImgVol);
        AuxInputImgVol.Free;
      end;
      OutputImgVol.Resize((MaxTileX + 1)*csTileSize*2, (MaxTileY + 1)*csTileSize*2, 3);
      WriteLn('Resizing with tiles to: ', OutputImgVol.SizeX,' x ', OutputImgVol.SizeY,' x ',OutputImgVol.Depth);
      OutputImgVol.Fill(0);
      for TileXCnt := 0 to MaxTileX do
      begin
        for TileYCnt := 0 to MaxTileY do
        begin
          TileStartX := TileXCnt*csTileSize;
          TileStartY := TileYCnt*csTileSize;
          InputTile.CopyCropping
          (
            InputImgVol,
            TileStartX,
            TileStartY,
            csTileSize+csTileBorder*2,
            csTileSize+csTileBorder*2
          );
          NN.Compute(InputTile);
          NN.GetOutput(OutputTile);
          OutputImgVol.AddArea
          (
            {DestX=}TileXCnt*csTileSize*2,
            {DestY=}TileYCnt*csTileSize*2,
            {OriginX=}csTileBorder*2,
            {OriginX=}csTileBorder*2,
            {LenX=}csTileSize*2,
            {LenY=}csTileSize*2,
            OutputTile
          );
        end;
      end;
      InputTile.Free;
      OutputTile.Free;
    end;
    OutputImgVol.Add(2);
    OutputImgVol.Mul(64);
    LoadVolumeIntoImage(OutputImgVol, Image);
    WriteLn('Saving output file: ', outputFile);
    if Not(Image.SaveToFile(outputFile)) then
    begin
      WriteLn('Saving has failed: ', outputFile);
    end;
    OutputImgVol.Free;
    InputImgVol.Free;
    Image.Free;
    NN.Free;
  end;

  constructor TSuperResolutionExample.Create(TheOwner: TComponent);
  begin
    inherited Create(TheOwner);
    StopOnException := True;
  end;

  destructor TSuperResolutionExample.Destroy;
  begin
    inherited Destroy;
  end;

  procedure TSuperResolutionExample.WriteHelp;
  begin
    WriteLn
    (
      'Increase Image Resolution from an image file',sLineBreak,
      'Command Line Example: SuperResolution -i myphoto.png -o myphoto-big.png', sLineBreak,
      ' -h : displays this help. ', sLineBreak,
      ' -i : defines input file. ', sLineBreak,
      ' -o : defines output file.', sLineBreak,
      ' More info at:', sLineBreak,
      '   https://github.com/joaopauloschuler/neural-api', sLineBreak
    );
  end;

var
  Application: TSuperResolutionExample;
begin
  Application := TSuperResolutionExample.Create(nil);
  Application.Title:='Super Resolution Command Line Example';
  Application.Run;
  Application.Free;
end.
