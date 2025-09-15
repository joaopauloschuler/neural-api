(*
neuralvolume
Copyright (C) 2016 Joao Paulo Schwarz Schuler

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

unit neuralhelper;
// Coded by achinastone

interface

uses classes, sysutils, neuralvolume;

{$include neuralnetwork.inc}

type
  TNeuralFloatDynArr = array of TNeuralFloat;

  TNNetVolumeHelper = class helper for TNNetVolume
  private
    procedure SetDataArr(const ADataArr: TNeuralFloatDynArr);
  public
    property DataArr: TNeuralFloatDynArr write SetDataArr;
  end;

implementation

procedure TNNetVolumeHelper.SetDataArr(const ADataArr: TNeuralFloatDynArr);
begin
  if Length(FData) <> Length(FData) then
    raise Exception.Create('DataArr length not matched ' + IntToStr(Length(FData)) + ' and ' + IntTostr(Length(FData)));

  Move(ADataArr[0], FData[0], SizeOf(TNeuralFloat) * Length(FData));
end;

end.
