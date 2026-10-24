/// Human-pose keypoint estimation with an imported ViTPose model.
///
/// ViTPose (Xu et al. 2022, "ViTPose: Simple Vision Transformer Baselines for
/// Human Pose Estimation") is a top-down pose estimator: a plain ViT backbone
/// runs over a crop around ONE person and a small deconvolution head turns the
/// patch-grid features into one 2-D heatmap per keypoint. This example runs the
/// whole top-down recipe of transformers' VitPoseImageProcessor:
///   person box -> padded crop warped to the model input -> ViTPose ->
///   heatmap peak + DARK refinement -> keypoints in photo pixels.
/// The person boxes come from --box (one box), from a YOLOv8 detector with
/// --yolo (every person in the photo), or default to the whole photo.
/// --openpose-out also writes the OpenPose image of the poses, drawn as
/// controlnet_aux DWpose does (DrawOpenPoseBodiesViaDetectResolution), the
/// control image of an openpose ControlNet.
///
/// With no arguments it runs the committed PICO fixtures (random weights, a
/// synthetic photo), so the numbers exercise the pipeline, not real joints.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program PoseEstimation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained,
  neuralimageresize;

type
  TPoseOptions = record
    ModelFile, ConfigFile, ImageFile, OutputFile: string;
    // --openpose-out: the OpenPose control image of the poses; '' for none.
    OpenPoseFile: string;
    // --detect-resolution: DWpose draw short side; 0 draws at the photo size.
    DetectResolution: integer;
    YoloFile, YoloConfigFile: string;
    HasBox: boolean;
    Box: array[0..3] of TNeuralFloat;
    ScoreThreshold: TNeuralFloat;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: PoseEstimation                          (pico fixture smoke)');
  WriteLn('       PoseEstimation --model FILE --image IMG [options]');
  WriteLn('  --model FILE         ViTPose model.safetensors');
  WriteLn('                       (usyd-community/vitpose-base-simple)');
  WriteLn('  --config FILE        ViTPose config.json (default: next to --model)');
  WriteLn('  --image IMG          the photo');
  WriteLn('  --box X,Y,W,H        one person box in photo pixels (top-left, size)');
  WriteLn('  --yolo FILE          YOLOv8 safetensors: one pose per person (class 0)');
  WriteLn('  --yolo-config FILE   YOLOv8 config.json (default: next to --yolo)');
  WriteLn('  --conf X             person score threshold (default 0.25)');
  WriteLn('  --output FILE        photo with boxes and joints (default pose_out.png)');
  WriteLn('  --openpose-out FILE  also the OpenPose image (ControlNet openpose input)');
  WriteLn('  --detect-resolution N');
  WriteLn('                       OpenPose draw short side, then resized to the photo');
  WriteLn('                       (default 512, as DWpose; 0 = draw at the photo size)');
  WriteLn('Without --box or --yolo the whole photo is one person box.');
end;

function FixturePath(const FileName: string): string;
begin
  Result := '../../tests/fixtures/' + FileName;
  if not FileExists(Result) then Result := 'tests/fixtures/' + FileName;
end;

function ParseOptions(out Opt: TPoseOptions): boolean;
var
  ArgPos: integer;
  Arg: string;
  Parts: TStringList;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

  procedure ParseBox(const Text: string);
  var
    PartPos: integer;
  begin
    Parts.DelimitedText := Text;
    if Parts.Count <> 4 then
      raise Exception.Create('--box expects X,Y,W,H, got "' + Text + '"');
    for PartPos := 0 to 3 do
      Opt.Box[PartPos] := StrToFloat(Trim(Parts[PartPos]));
    Opt.HasBox := true;
  end;

begin
  Result := true;
  Opt.ModelFile := ''; Opt.ConfigFile := ''; Opt.ImageFile := '';
  Opt.OutputFile := 'pose_out.png';
  Opt.OpenPoseFile := '';
  Opt.DetectResolution := 512;
  Opt.YoloFile := ''; Opt.YoloConfigFile := '';
  Opt.HasBox := false;
  Opt.ScoreThreshold := 0.25;
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ',';
    Parts.StrictDelimiter := true;
    ArgPos := 1;
    while ArgPos <= ParamCount do
    begin
      Arg := ParamStr(ArgPos);
      if Arg = '--model' then Opt.ModelFile := NextArg()
      else if Arg = '--config' then Opt.ConfigFile := NextArg()
      else if Arg = '--image' then Opt.ImageFile := NextArg()
      else if Arg = '--box' then ParseBox(NextArg())
      else if Arg = '--yolo' then Opt.YoloFile := NextArg()
      else if Arg = '--yolo-config' then Opt.YoloConfigFile := NextArg()
      else if Arg = '--conf' then Opt.ScoreThreshold := StrToFloat(NextArg())
      else if Arg = '--output' then Opt.OutputFile := NextArg()
      else if Arg = '--openpose-out' then Opt.OpenPoseFile := NextArg()
      else if Arg = '--detect-resolution' then
        Opt.DetectResolution := StrToInt(NextArg())
      else if (Arg = '--help') or (Arg = '-h') then
      begin
        PrintUsage();
        exit(false);
      end
      else
        raise Exception.Create('unknown argument: ' + Arg);
      Inc(ArgPos);
    end;
  finally
    Parts.Free;
  end;
  if (Opt.ModelFile = '') <> (Opt.ImageFile = '') then
    raise Exception.Create('--model and --image go together');
  if Opt.HasBox and (Opt.YoloFile <> '') then
    raise Exception.Create('use --box or --yolo, not both');
end;

procedure PrintPose(const Pose: TPoseKeypoints);
var
  Joint: integer;
begin
  WriteLn(Format('person box (%.1f, %.1f)-(%.1f, %.1f) score %.3f',
    [Pose.X1, Pose.Y1, Pose.X2, Pose.Y2, Pose.BoxScore]));
  for Joint := 0 to High(Pose.Joints) do
    WriteLn(Format('  joint %2d: x=%8.2f y=%8.2f score=%.4f',
      [Joint, Pose.Joints[Joint].X, Pose.Joints[Joint].Y,
       Pose.Joints[Joint].Score]));
end;

// Draws each box outline in green with a "person score" tag above it and each
// joint as a red square.
procedure DrawPoses(Image: TNNetVolume; const Poses: TPoseKeypointsArray);
const
  csBoxColor: array[0..2] of TNeuralFloat = (0, 255, 0);
  csTextColor: array[0..2] of TNeuralFloat = (0, 0, 0);
  csJointColor: array[0..2] of TNeuralFloat = (255, 0, 0);
var
  PersonPos, Joint, Radius, TextScale, TagHeight, TagTop, X1, Y1, X2, Y2,
    JointX, JointY: integer;
  BoxLabel: string;
begin
  Radius := Max(1, Min(Image.SizeX, Image.SizeY) div 200);
  TextScale := Max(1, Min(Image.SizeX, Image.SizeY) div 240);
  TagHeight := (csBitmapFontGlyphHeight + 2) * TextScale;
  for PersonPos := 0 to High(Poses) do
  begin
    X1 := Round(Poses[PersonPos].X1);
    Y1 := Round(Poses[PersonPos].Y1);
    X2 := Round(Poses[PersonPos].X2);
    Y2 := Round(Poses[PersonPos].Y2);
    DrawRectangleOutline(Image, X1, Y1, X2, Y2, csBoxColor);
    BoxLabel := Format('person %.2f', [Poses[PersonPos].BoxScore]);
    TagTop := Y1 - TagHeight;
    if TagTop < 0 then TagTop := Y1;
    FillRectangleCV2(Image, X1, TagTop,
      X1 + BitmapTextWidth(BoxLabel, TextScale) + 2 * TextScale - 1,
      TagTop + TagHeight - 1, csBoxColor);
    DrawBitmapText(Image, X1 + TextScale, TagTop + TextScale, BoxLabel,
      csTextColor, TextScale);
    for Joint := 0 to High(Poses[PersonPos].Joints) do
    begin
      JointX := Round(Poses[PersonPos].Joints[Joint].X);
      JointY := Round(Poses[PersonPos].Joints[Joint].Y);
      FillRectangleCV2(Image, JointX - Radius, JointY - Radius, JointX + Radius,
        JointY + Radius, csJointColor);
    end;
  end;
end;

// Runs ViTPose on one box (or YOLO boxes when YoloNN is not nil) of Image.
function EstimatePoses(PoseNN: TNNet; const PoseConfig: TViTPoseConfig;
  YoloNN: TNNet; const YoloConfig: TYoloConfig; Image: TNNetVolume;
  const Opt: TPoseOptions): TPoseKeypointsArray;
begin
  if YoloNN <> nil then
    exit(EstimateMultiPersonPose(YoloNN, YoloConfig, PoseNN, PoseConfig, Image,
      Opt.ScoreThreshold));
  SetLength(Result, 1);
  if Opt.HasBox then
  begin
    Result[0].X1 := Opt.Box[0];
    Result[0].Y1 := Opt.Box[1];
    Result[0].X2 := Opt.Box[0] + Opt.Box[2];
    Result[0].Y2 := Opt.Box[1] + Opt.Box[3];
  end
  else
  begin
    Result[0].X1 := 0;
    Result[0].Y1 := 0;
    Result[0].X2 := Image.SizeX;
    Result[0].Y2 := Image.SizeY;
  end;
  Result[0].BoxScore := 1;
  Result[0].Joints := EstimateViTPoseInBox(PoseNN, PoseConfig, Image,
    Result[0].X1, Result[0].Y1, Result[0].X2 - Result[0].X1,
    Result[0].Y2 - Result[0].Y1);
end;

var
  Opt: TPoseOptions;
  PoseNN, YoloNN: TNNet;
  PoseConfig: TViTPoseConfig;
  YoloConfig: TYoloConfig;
  Image, OpenPoseImage: TNNetVolume;
  Poses: TPoseKeypointsArray;
  X, Y, Channel, PersonPos: integer;
  IsSmoke: boolean;
begin
  DefaultFormatSettings.DecimalSeparator := '.';
  if not ParseOptions(Opt) then exit;
  IsSmoke := Opt.ModelFile = '';
  if IsSmoke then
  begin
    WriteLn('Pico fixture smoke: random weights, synthetic photo.');
    Opt.ModelFile := FixturePath('tiny_vitpose.safetensors');
    Opt.ConfigFile := FixturePath('tiny_vitpose_config.json');
    Opt.YoloFile := FixturePath('tiny_yolo.safetensors');
    Opt.YoloConfigFile := FixturePath('tiny_yolo_config.json');
    Opt.ScoreThreshold := 0.53;
  end;
  YoloNN := nil;
  PoseNN := BuildViTPoseFromSafeTensors(Opt.ModelFile, PoseConfig,
    {pTrainable=}false, Opt.ConfigFile);
  Image := TNNetVolume.Create;
  OpenPoseImage := TNNetVolume.Create;
  try
    WriteLn(ViTPoseConfigToString(PoseConfig));
    if Opt.YoloFile <> '' then
      YoloNN := BuildYoloFromSafeTensors(Opt.YoloFile, YoloConfig,
        {pTrainable=}false, Opt.YoloConfigFile);
    if IsSmoke then
    begin
      Image.ReSize(150, 90, 3);
      for Y := 0 to Image.SizeY - 1 do
        for X := 0 to Image.SizeX - 1 do
          for Channel := 0 to 2 do
            Image[X, Y, Channel] := Round(127.5 + 127.5 *
              Sin(X * 0.11 + Y * 0.07 * (Channel + 1) + Channel));
    end
    else if not LoadImageFromFileIntoVolume(Opt.ImageFile, Image) then
      raise Exception.Create('cannot read ' + Opt.ImageFile);
    Poses := EstimatePoses(PoseNN, PoseConfig, YoloNN, YoloConfig, Image, Opt);
    WriteLn(Length(Poses), ' person(s) on the ', Image.SizeX, 'x', Image.SizeY,
      ' photo.');
    for PersonPos := 0 to High(Poses) do PrintPose(Poses[PersonPos]);
    if Opt.OpenPoseFile <> '' then
    begin
      if Opt.DetectResolution > 0 then
        DrawOpenPoseBodiesViaDetectResolution(OpenPoseImage, Image.SizeX,
          Image.SizeY, Poses, Opt.DetectResolution)
      else
        DrawOpenPoseBodies(OpenPoseImage, Image.SizeX, Image.SizeY, Poses);
      if SaveImageFromVolumeIntoFile(OpenPoseImage, Opt.OpenPoseFile) then
        WriteLn('Wrote ', Opt.OpenPoseFile)
      else
        WriteLn('Could not write ', Opt.OpenPoseFile);
    end;
    if not IsSmoke then
    begin
      DrawPoses(Image, Poses);
      if SaveImageFromVolumeIntoFile(Image, Opt.OutputFile) then
        WriteLn('Wrote ', Opt.OutputFile)
      else
        WriteLn('Could not write ', Opt.OutputFile);
    end;
  finally
    OpenPoseImage.Free;
    Image.Free;
    YoloNN.Free;
    PoseNN.Free;
  end;
end.
