program SuperResolution;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api/tree/master/examples/SuperResolution
 This command line tool increases image resolution from an input image file.
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
  neuralgeneric,
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
    csTileBorder = 4;

  procedure TSuperResolutionExample.RunAlgo(inputFile, outputFile: string);
  var
    NN: TNNet;
    InputImgVol, OutputImgVol: TNNetVolume;
    InputTile, OutputTile: TNNetVolume;
    MaxTileX, MaxTileY: integer;
    TileXCnt, TileYCnt: integer;
    TileStartX, TileStartY: integer;
    LocalDestX, LocalDestY: integer;
    LocalOriginX, LocalOriginY: integer;
    LocalLenX, LocalLenY: integer;
    LocalTileSizeX, LocalTileSizeY: integer;
  begin
    InputImgVol := TNNetVolume.Create();
    OutputImgVol := TNNetVolume.Create();
    WriteLn('Loading input file: ', inputFile);
    LoadImageFromFileIntoVolume(inputFile, InputImgVol);
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
      LocalTileSizeX := GetMaxDivisor(InputImgVol.SizeX-csTileBorder*2, 128);
      LocalTileSizeY := GetMaxDivisor(InputImgVol.SizeY-csTileBorder*2, 128);
      WriteLn('Resizing with tiles. Tile size is: ', LocalTileSizeX, 'x', LocalTileSizeY, ' .');
      NN := CreateResizingNN(LocalTileSizeX+csTileBorder*2, LocalTileSizeY+csTileBorder*2, csExampleFileName);
      InputTile := TNNetVolume.Create(LocalTileSizeX+csTileBorder*2, LocalTileSizeY+csTileBorder*2, 3);
      OutputTile := TNNetVolume.Create(InputTile.SizeX*2, InputTile.SizeY*2, 3);
      MaxTileX := ( (InputImgVol.SizeX-csTileBorder*2) div LocalTileSizeX) - 1;
      MaxTileY := ( (InputImgVol.SizeY-csTileBorder*2) div LocalTileSizeY) - 1;
      OutputImgVol.Resize(InputImgVol.SizeX*2, InputImgVol.SizeY*2, 3);
      WriteLn('Resizing with tiles to: ', OutputImgVol.SizeX,' x ', OutputImgVol.SizeY,' x ',OutputImgVol.Depth);
      OutputImgVol.Fill(0);
      for TileXCnt := 0 to MaxTileX do
      begin
        for TileYCnt := 0 to MaxTileY do
        begin
          TileStartX := TileXCnt*LocalTileSizeX;
          TileStartY := TileYCnt*LocalTileSizeY;
          InputTile.CopyCropping
          (
            InputImgVol,
            TileStartX,
            TileStartY,
            LocalTileSizeX+csTileBorder*2,
            LocalTileSizeY+csTileBorder*2
          );
          NN.Compute(InputTile);
          NN.GetOutput(OutputTile);
          LocalDestX := TileXCnt*LocalTileSizeX*2+csTileBorder*2;
          LocalDestY := TileYCnt*LocalTileSizeY*2+csTileBorder*2;
          LocalOriginX := csTileBorder*2;
          LocalOriginY := csTileBorder*2;
          LocalLenX := LocalTileSizeX*2;
          LocalLenY := LocalTileSizeY*2;

          if ((TileXCnt = 0) or (TileXCnt = MaxTileX)) then
          begin
            LocalLenX := LocalLenX + csTileBorder*2;
            if (TileXCnt = 0) then
            begin
              LocalOriginX := 0;
              LocalDestX := 0;
            end;
          end;

          if ((TileYCnt = 0) or (TileYCnt = MaxTileY)) then
          begin
            LocalLenY := LocalLenY + csTileBorder*2;
            if (TileYCnt = 0) then
            begin
              LocalOriginY := 0;
              LocalDestY := 0;
            end;
          end;

          OutputImgVol.AddArea
          (
            {DestX=}LocalDestX,
            {DestY=}LocalDestY,
            {OriginX=}LocalOriginX,
            {OriginY=}LocalOriginY,
            {LenX=}LocalLenX,
            {LenY=}LocalLenY,
            OutputTile
          );
        end;
      end;
      InputTile.Free;
      OutputTile.Free;
    end;
    OutputImgVol.Add(2);
    OutputImgVol.Mul(64);
    WriteLn('Saving output file: ', outputFile);
    if Not(SaveImageFromVolumeIntoFile(OutputImgVol, outputFile)) then
    begin
      WriteLn('Saving has failed: ', outputFile);
    end;
    OutputImgVol.Free;
    InputImgVol.Free;
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
      '   https://github.com/joaopauloschuler/neural-api/tree/master/examples/SuperResolution', sLineBreak
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
