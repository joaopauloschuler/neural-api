program GumbelSoftmaxDemo;
(*
GumbelSoftmaxDemo: demonstrates TNNetGumbelSoftmax — a differentiable
categorical sampling head computing y = softmax((logits + g) / tau) where
g ~ Gumbel(0,1).

Part (a): on a FIXED logit vector it sweeps the temperature tau across
{2.0, 1.0, 0.5, 0.1} on the deterministic inference path (no noise) and prints
the resulting softmax distribution plus its Shannon entropy. As tau shrinks the
distribution sharpens towards a one-hot vector (entropy falls).

Part (b): shows that hard straight-through mode emits a one-hot output.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  SysUtils,
  neuralnetwork,
  neuralvolume;

const
  NumClasses = 5;
  Logits: array[0..NumClasses - 1] of TNeuralFloat =
    (2.5, 1.0, 0.3, -0.5, 0.8);
  Taus: array[0..3] of TNeuralFloat = (2.0, 1.0, 0.5, 0.1);

procedure FillLogits(AInput: TNNetVolume);
var
  i: integer;
begin
  for i := 0 to NumClasses - 1 do
    AInput.Raw[i] := Logits[i];
end;

function Entropy(AOutput: TNNetVolume): TNeuralFloat;
var
  i: integer;
  p: TNeuralFloat;
begin
  Result := 0;
  for i := 0 to AOutput.Size - 1 do
  begin
    p := AOutput.Raw[i];
    if p > 1e-12 then Result := Result - p * Ln(p);
  end;
end;

procedure RunSoftSweep();
var
  NN: TNNet;
  Input: TNNetVolume;
  TauIdx, i: integer;
  Line: string;
begin
  WriteLn('(a) Soft mode tau sweep on a fixed logit vector (inference path).');
  Write('    logits           = ');
  for i := 0 to NumClasses - 1 do Write(Format('%8.4f', [Logits[i]]));
  WriteLn;
  WriteLn('    As tau shrinks the distribution sharpens and entropy falls.');
  WriteLn;
  for TauIdx := 0 to High(Taus) do
  begin
    NN := TNNet.Create();
    Input := TNNetVolume.Create(1, 1, NumClasses);
    try
      NN.AddLayer(TNNetInput.Create(1, 1, NumClasses, 1));
      // Soft mode (hard=0). The layer is disabled by default, so Compute uses
      // the deterministic inference path: y = softmax(logits / tau).
      NN.AddLayer(TNNetGumbelSoftmax.Create(Taus[TauIdx], 0));
      FillLogits(Input);
      NN.Compute(Input);
      Line := '';
      for i := 0 to NumClasses - 1 do
        Line := Line + Format('%8.4f', [NN.GetLastLayer.Output.Raw[i]]);
      WriteLn(Format('    tau=%5.2f  y = %s  entropy=%6.4f',
        [Taus[TauIdx], Line, Entropy(NN.GetLastLayer.Output)]));
    finally
      NN.Free;
      Input.Free;
    end;
  end;
  WriteLn;
end;

procedure RunHardDemo();
var
  NN: TNNet;
  Input: TNNetVolume;
  i: integer;
  Line: string;
begin
  WriteLn('(b) Hard straight-through mode emits a one-hot vector.');
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1, 1, NumClasses);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, NumClasses, 1));
    NN.AddLayer(TNNetGumbelSoftmax.Create(1.0, 1));
    FillLogits(Input);
    NN.Compute(Input);
    Line := '';
    for i := 0 to NumClasses - 1 do
      Line := Line + Format('%8.4f', [NN.GetLastLayer.Output.Raw[i]]);
    WriteLn('    hard y = ' + Line + '  (one-hot at the argmax logit, class 0)');
  finally
    NN.Free;
    Input.Free;
  end;
  WriteLn;
end;

begin
  WriteLn('TNNetGumbelSoftmax demo: differentiable categorical sampling.');
  WriteLn('y = softmax((logits + g) / tau),  g ~ Gumbel(0,1).');
  WriteLn;
  RunSoftSweep();
  RunHardDemo();
  WriteLn('Done.');
end.
