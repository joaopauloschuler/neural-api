(*
neuraldatasetsv
Copyright (C) 2018 Joao Paulo Schwarz Schuler

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

unit neuraldatasetsv;

{$mode objfpc}{$H+}

interface

uses
  neuraldatasets, Classes, SysUtils, ExtCtrls, Graphics,
  neuralvolume;

{$IFDEF FPC}
type
  { TClassesAndElements }
  TClassesAndElements = class(TStringStringListVolume)
    private
      FImageSubFolder: string;
      FBaseFolder: string;
    public
      constructor Create();
      function CountElements(): integer;
      procedure LoadFoldersAsClasses(FolderName: string; pImageSubFolder: string = ''; SkipFirst: integer = 0; SkipLast: integer = 0);
      procedure LoadImages(color_encoding: integer);
      procedure LoadClass_FilenameFromFolder(FolderName: string);
      function GetRandomClassId(): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetFileName(ClassId, ElementId: integer): string; {$IFDEF Release} inline; {$ENDIF}
  end;
{$ENDIF}

// Loads a TinyImage into FCL TImage.
procedure LoadTinyImageIntoTImage(var TI: TTinyImage; var Image: TImage);

// Loads a single channel tiny image into image.
procedure LoadTISingleChannelIntoImage(var TI: TTinySingleChannelImage;
  var Image: TImage);

implementation

uses neuralvolumev, fileutil;

procedure LoadTinyImageIntoTImage(var TI: TTinyImage; var Image: TImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Image.Canvas.Pixels[J, I] := RGBToColor(TI.R[I, J], TI.G[I, J], TI.B[I, J]);
    end;
  end;
end;

procedure LoadTISingleChannelIntoImage(var TI: TTinySingleChannelImage;
  var Image: TImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Image.Canvas.Pixels[J, I] :=
        RGBToColor(TI.Grey[I, J], TI.Grey[I, J], TI.Grey[I, J]);
    end;
  end;
end;

{$IFDEF FPC}
{ TClassesAndElements }
constructor TClassesAndElements.Create();
begin
  inherited Create();
  FImageSubFolder := '';
  FBaseFolder := '';
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
      Result += Self.List[ClassId].Count;
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
          ClassFolder += FImageSubFolder + DirectorySeparator;
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

procedure TClassesAndElements.LoadImages(color_encoding: integer);
var
  LocalPicture: TPicture;
  SourceVolume: TNNetVolume;
  ClassId, ImageId: integer;
  MaxClass, MaxImage: integer;
begin
  LocalPicture := TPicture.Create;
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
          LocalPicture.LoadFromFile( Self.GetFileName(ClassId, ImageId) );
          LoadPictureIntoVolume(LocalPicture, SourceVolume);
          SourceVolume.Tag := ClassId;
          SourceVolume.RgbImgToNeuronalInput(color_encoding);
        end;
      end;
    end;
  end;
  LocalPicture.Free;
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

function TClassesAndElements.GetFileName(ClassId, ElementId: integer): string;
begin
  Result := Self.List[ClassId].Strings[ElementId];
end;
{$ENDIF}


end.

