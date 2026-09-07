unit neuraldelphi;

interface

procedure FillDWord(var X; Count: NativeUInt; Value: Cardinal);

implementation

procedure FillDWord(var X; Count: NativeUInt; Value: Cardinal);
var
  P: PCardinal;
  I: NativeUInt;
  CountM1: NativeUInt;
begin
  P := @X;
  CountM1 := Count - 1;
  for I := 0 to CountM1 do
  begin
    P^ := Value;
    Inc(P);
  end;
end;

end.
