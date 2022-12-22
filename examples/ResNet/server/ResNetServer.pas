program ResNetServer;

{$mode objfpc}{$H+}

uses
  {$ifdef unix}cthreads, {$endif}
  Interfaces, // this includes the LCL widgetset
  Classes, SysUtils, FileUtil, Graphics, GraphUtil,
  neuralnetwork, neuralvolume, neuraldatasets, neuralthread,
  fphttpapp, httpdefs, httproute, fpjson;

procedure LoadBitmapIntoTinyImage(var Bitmap: TBitmap; out TI: TTinyImage);
var
  I, J: integer;
  r,g,b: byte;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      RedGreenBlue(Bitmap.Canvas.Pixels[J, I], r,g,b);
      TI.R[I, J] := r;
      TI.G[I, J] := g;
      TI.B[I, J] := b;
    end;
  end;
end;

procedure Home(aRequest: TRequest; aResponse: TResponse);
begin
  aResponse.ContentType := 'text/html';
  with aResponse.Contents do
  begin
    Add('<html><body>');
    Add('<form action="/nn" method="post" enctype="multipart/form-data">');
    Add('<input type="file" name="input" />');
    Add('<br/><br/>');
    Add('<fieldset style="border: 0">'+
    '<legend>Select the output type:</legend>'+
    '<div>'+
    '  <input type="radio" id="text" name="otype" value="text" checked>'+
    '  <label for="text">text</label>'+
    '</div>'+
    '<div>'+
    '  <input type="radio" id="json" name="otype" value="json">'+
    '  <label for="huey">json</label>'+
    '</div>'+
'</fieldset>');
    Add('<br/>');
    Add('<input type="submit" value="Classify" />');
    Add('</form>');
    Add('</body></html>');
  end;
  aResponse.SendContent;
end;


var
  NN: TNNet;
  Labels: TStringList;
  FCritSec: TRTLCriticalSection;

procedure Endpoint(aRequest: TRequest; aResponse: TResponse);
var
  jObject : TJSONObject;
  SrcImage: TPicture;
  PredImage: TBitmap;
  TI: TTinyImage;
  i: integer;
  InputV, OutputV: TNNetVolume;
  LocalNN: TNNet;
  OutputType, OutputStr: string;
  FoundClassId: integer;
  OutputStrLst: TStringList;
begin
  SrcImage := TPicture.Create;
  PredImage := TBitmap.Create;
  InputV := TNNetVolume.Create;
  OutputV := TNNetVolume.Create;
  LocalNN := NN;
  PredImage.SetSize(32,32);
  OutputType := ARequest.ContentFields.Values['otype'];

  try
    SrcImage.LoadFromStream(ARequest.Files[0].Stream);
    PredImage.Canvas.StretchDraw(TRect.Create(0,0,32,32), SrcImage.Bitmap);
    {PredImage.SaveToFile('/tmp/tmp.bmp');}
    LoadBitmapIntoTinyImage(PredImage, TI);
    LoadTinyImageIntoNNetVolume(TI, InputV);
    // Bipolar representation.
    InputV.RgbImgToNeuronalInput(csEncodeRGB);
  finally
    SrcImage.Free;
    PredImage.Free;
  end;

  // Critical Section - Enter
  EnterCriticalSection(FCritSec);
  try
    LocalNN.Compute(InputV);
    LocalNN.GetOutput(OutputV);
    // Increases accuracy with a flipped version.
    InputV.FlipX();
    LocalNN.Compute(InputV);
    LocalNN.AddOutput(OutputV);
  finally
    // Critical Section - Exit
    LeaveCriticalSection(FCritSec);
  end;

  OutputV.Divi(2);

  if (OutputType = 'text') then
  begin
    FoundClassId := OutputV.GetClass();
    OutputStrLst := TStringList.Create;
    OutputStrLst.Delimiter := ' ';
    OutputStrLst.Add('<html><body>'+
      'The found class is: '+Labels.ValueFromIndex[FoundClassId]+'.<br/><br/>');
    for i := 0 to OutputV.Size - 1 do
    begin
      OutputStr := '';
      if i = FoundClassId then OutputStr += '<b>';
      OutputStr += Labels.ValueFromIndex[i]+': '+FloatToStr(Round(OutputV.Raw[i]*10000)/100)+'%.<br/>';
      if i = FoundClassId then OutputStr += '</b>';
      OutputStrLst.Add(OutputStr);
    end;
    OutputStrLst.Add('</body></html>');
    aResponse.ContentType := 'text/html';
    aResponse.Content := OutputStrLst.Text;
    OutputStrLst.Free;
  end
  else
  begin
    jObject := TJSONObject.Create;
    try
      for i := 0 to OutputV.Size - 1 do
      begin
        jObject.Floats[Labels.ValueFromIndex[i]] := OutputV.Raw[i];
      end;
      aResponse.Content := jObject.AsJSON;
      aResponse.ContentType := 'application/json';
    finally
      jObject.Free;
    end;
  end;
  aResponse.SendContent;

  InputV.Free;
  OutputV.Free;
end;


var
  LabelFile: TextFile;
  line: string;

begin
   if paramCount <> 3 then
   begin
     WriteLn('Usage: ResNetServer <port> <model> <labels>');
     exit;
   end;
   NeuralInitCriticalSection(FCritSec);
   NN := TNNet.Create;
   NN.LoadFromFile(paramStr(2));
   NN.EnableDropouts(false);

   Labels := TStringList.Create;
   AssignFile(LabelFile, paramStr(3));
   try
     reset(LabelFile);
     repeat
       ReadLn(LabelFile, line);
       Labels.add(line);
     until eof(LabelFile);
   finally
     Close(LabelFile);
   end;

   WriteLn(Labels.Count,' class labels loaded.');
   if NN.GetLastLayer().Output.Size <> Labels.Count then
   begin
     WriteLn('WARNING: you have ',Labels.Count,' lables and ',
     NN.GetLastLayer().Output.Size,' outputs on the last layer.');
   end;
   HTTPRouter.RegisterRoute('/nn', @Endpoint);
   HTTPRouter.RegisterRoute('/', @Home);
   Application.Port := StrToInt(paramStr(1));
   Application.Threaded := true;
   Application.Initialize;
   WriteLn('Listening in port ', paramStr(1));
   Application.Run;
   Labels.Free;
   NN.Free;
   NeuralDoneCriticalSection(FCritSec);
   WriteLn('Bye bye.');
end.

