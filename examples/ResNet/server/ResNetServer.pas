program ResNetServer;

{$mode objfpc}{$H+}

uses
  {$ifdef unix}cthreads, {$endif}
  Interfaces, // this includes the LCL widgetset
  Classes, SysUtils, FileUtil, Graphics, GraphUtil,
  neuralnetwork, neuralvolume, neuraldatasets,
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
    Add('<input type="submit" value="Classify" />');
    Add('</form>');
    Add('</body></html>');
  end;
  aResponse.SendContent;
end;


var
  NN: TNNet;
  Labels: TStringList;

procedure Endpoint(aRequest: TRequest; aResponse: TResponse);
var
  jObject : TJSONObject;
  SrcImage: TPicture;
  PredImage: TBitmap;
  TI: TTinyImage;
  i: integer;
  InputV, OutputV: TNNetVolume;
  LocalNN: TNNet;
begin
  SrcImage := TPicture.Create;
  PredImage := TBitmap.Create;
  InputV := TNNetVolume.Create;
  OutputV := TNNetVolume.Create;
  LocalNN := NN.Clone();
  PredImage.SetSize(32,32);

  try
    SrcImage.LoadFromStream(ARequest.Files[0].Stream);
    PredImage.Canvas.StretchDraw(TRect.Create(0,0,32,32), SrcImage.Bitmap);
    {PredImage.SaveToFile('/tmp/tmp.bmp');}
    LoadBitmapIntoTinyImage(PredImage, TI);
    LoadTinyImageIntoNNetVolume(TI, InputV);
    // Bipolar representation.
    InputV.Divi(64);
    InputV.Sub(2);
  finally
    SrcImage.Free;
    PredImage.Free;
  end;

  LocalNN.Compute(InputV);
  LocalNN.GetOutput(OutputV);

  // Increases accuracy with a flipped version.
  InputV.FlipX();
  LocalNN.Compute(InputV);
  LocalNN.AddOutput(OutputV);

  OutputV.Divi(2);

  jObject := TJSONObject.Create;
  try
    for i := 0 to OutputV.SizeX - 1 do
    begin
      jObject.Floats[Labels.ValueFromIndex[i]] := OutputV.Raw[i];
    end;
    aResponse.Content := jObject.AsJSON;
    aResponse.ContentType := 'application/json';
    aResponse.SendContent;
  finally
    jObject.Free;
  end;
  InputV.Free;
  OutputV.Free;
  LocalNN.Free;
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
   NN := TNNet.Create;
   NN.LoadFromFile(paramStr(2));

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

   HTTPRouter.RegisterRoute('/nn', @Endpoint);
   HTTPRouter.RegisterRoute('/', @Home);
   Application.Port := StrToInt(paramStr(1));
   Application.Threaded := true;
   Application.Initialize;
   WriteLn('Listening in port ', paramStr(1));
   Application.Run;
   Labels.Free;
   NN.Free;
   WriteLn('Bye bye.');
end.

