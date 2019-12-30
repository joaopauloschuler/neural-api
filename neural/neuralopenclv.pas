(*
neuralopenclv
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

unit neuralopenclv;

{$IFDEF FPC} {$mode objfpc}{$H+} {$ENDIF}

interface

uses
  Classes, SysUtils, neuralopencl, StdCtrls;

type
  { TEasyOpenCLCL }
  TEasyOpenCLCL = class(TEasyOpenCLV)
    public
      constructor Create();
      destructor Destroy(); override;

      function ProbeClick(ComboPlatform, ComboDevType: TComboBox): boolean;
      procedure ComboPlatformChange(ComboPlatform, ComboDevType: TComboBox);
      procedure ComboDevTypeChange(ComboDevType: TComboBox);
  end;


implementation
uses cl;

{ TEasyOpenCLCL }
constructor TEasyOpenCLCL.Create;
begin
  inherited Create();
end;

destructor TEasyOpenCLCL.Destroy;
begin
  inherited Destroy;
end;

function TEasyOpenCLCL.ProbeClick(ComboPlatform,
  ComboDevType: TComboBox): boolean;
var
  Platforms: TPlatformNames;
  I: integer;
begin
  Result := False;
  ComboPlatform.Items.Clear();
  Platforms := PlatformNames;
  if Length(Platforms) > 0 then
  begin
    for I := low(Platforms) to high(Platforms) do
    begin
      ComboPlatform.Items.Add(Platforms[I]);
    end;
    ComboPlatform.ItemIndex := 0;
    ComboPlatformChange(ComboPlatform, ComboDevType);

    ComboPlatform.Enabled := True;
    ComboDevType.Enabled := True;
    Result := True;
  end;
end;

procedure TEasyOpenCLCL.ComboPlatformChange(ComboPlatform,
  ComboDevType: TComboBox);
var
  PlatformId: cl_platform_id;
  I: integer;
begin
  PlatformId := PlatformIds[ComboPlatform.ItemIndex];
  SetCurrentPlatform(PlatformId);
  ComboDevType.Items.Clear();
  if GetDeviceCount()>0 then
  begin
    for I := 0 to GetDeviceCount() - 1 do
    begin
      ComboDevType.Items.Add(DeviceNames[I]);
    end;
    ComboDevType.ItemIndex := 0;
    ComboDevTypeChange(ComboDevType);
  end;
end;

procedure TEasyOpenCLCL.ComboDevTypeChange(ComboDevType: TComboBox);
var
  DeviceId: cl_device_id;
begin
  DeviceId := Devices[ComboDevType.ItemIndex];
  SetCurrentDevice(DeviceId);
end;

end.

