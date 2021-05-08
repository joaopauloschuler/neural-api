program Cifar10Resize;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread, usuperresolutionexample;

type
  TCifar10Resize = class(TCustomApplication)
  protected
    FNN32to64: THistoricalNets;
    FNN64to128: THistoricalNets;
    FNN128to256: THistoricalNets;
    FAux64Vol: TNNetVolume;
    FAux128Vol: TNNetVolume;
    FAux256Vol: TNNetVolume;
    procedure DoRun; override;
    procedure ProcessVolumeList(VolumeList: TNNetVolumeList; FolderName: string; BaseCount: integer);
  end;

  procedure CreateDirectories(FolderName: string; ClassCount:integer);
  var
    ClassCnt: integer;
  begin
    for ClassCnt := 0 to ClassCount - 1 do
    begin
      ForceDirectories(FolderName+'/class'+IntToStr(ClassCnt));
    end;
  end;

  procedure TCifar10Resize.ProcessVolumeList(VolumeList: TNNetVolumeList; FolderName: string; BaseCount: integer);
  var
    ImgCnt: integer;
    CurrentImg: TNNetVolume;
    ClassId: integer;
  begin
    for ImgCnt := 0 to VolumeList.Count - 1 do
    begin
      if ImgCnt mod 100 = 0 then
      begin
        WriteLn('Processing ', BaseCount + ImgCnt,' ', FolderName,'.');
      end;
      CurrentImg := VolumeList[ImgCnt];
      ClassId := CurrentImg.Tag;
      FNN32to64.Compute(CurrentImg);
      FNN32to64.GetOutput(FAux64Vol);
      FNN64to128.Compute(FAux64Vol);
      FNN64to128.GetOutput(FAux128Vol);
      FNN128to256.Compute(FAux128Vol);
      FNN128to256.GetOutput(FAux256Vol);
      CurrentImg.Add(2);
      CurrentImg.Mul(64);
      FAux64Vol.Add(2);
      FAux64Vol.Mul(64);
      FAux128Vol.Add(2);
      FAux128Vol.Mul(64);
      FAux256Vol.Add(2);
      FAux256Vol.Mul(64);
      SaveImageFromVolumeIntoFile(CurrentImg,
        'resized/cifar10-32/'+FolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(BaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(FAux64Vol,
        'resized/cifar10-64/'+FolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(BaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(FAux128Vol,
        'resized/cifar10-128/'+FolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(BaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(FAux256Vol,
        'resized/cifar10-256/'+FolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(BaseCount+ImgCnt)+'.png');
    end;
  end;

  procedure TCifar10Resize.DoRun;
  var
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Networks...');
    FNN32to64 := CreateResizingNN(32, 32, csExampleFileName);
    FNN64to128 := CreateResizingNN(64, 64, csExampleFileName);
    FNN128to256 := CreateResizingNN(128, 128, csExampleFileName);
    FAux64Vol := TNNetVolume.Create();
    FAux128Vol := TNNetVolume.Create();
    FAux256Vol := TNNetVolume.Create();
    CreateDirectories('resized/cifar10-32/train',10);
    CreateDirectories('resized/cifar10-64/train',10);
    CreateDirectories('resized/cifar10-128/train',10);
    CreateDirectories('resized/cifar10-256/train',10);
    CreateDirectories('resized/cifar10-32/train',10);
    CreateDirectories('resized/cifar10-64/train',10);
    CreateDirectories('resized/cifar10-128/train',10);
    CreateDirectories('resized/cifar10-256/train',10);
    CreateDirectories('resized/cifar10-32/test',10);
    CreateDirectories('resized/cifar10-64/test',10);
    CreateDirectories('resized/cifar10-128/test',10);
    CreateDirectories('resized/cifar10-256/test',10);
    CreateDirectories('resized/cifar10-32/test',10);
    CreateDirectories('resized/cifar10-64/test',10);
    CreateDirectories('resized/cifar10-128/test',10);
    CreateDirectories('resized/cifar10-256/test',10);

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    ProcessVolumeList(ImgTrainingVolumes, 'train', 0);
    ProcessVolumeList(ImgValidationVolumes, 'train', 40000);
    ProcessVolumeList(ImgTestVolumes, 'test', 0);

    FAux256Vol.Free;
    FAux128Vol.Free;
    FAux64Vol.Free;
    FNN128to256.Free;
    FNN64to128.Free;
    FNN32to64.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TCifar10Resize;
begin
  Application := TCifar10Resize.Create(nil);
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
