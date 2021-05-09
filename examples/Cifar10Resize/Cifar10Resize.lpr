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
    FFolderName: string;
    FBaseCount: integer;
    FThreadList: TNeuralThreadList;
    FCurrentVolumeList: TNNetVolumeList;
    FCriticalSection: TRTLCriticalSection;
    procedure DoRun; override;
    procedure ProcessVolumeList(VolumeList: TNNetVolumeList; FolderName: string; BaseCount: integer);
    procedure ResizeImages_NTL(index, threadnum: integer);
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
  begin
    FFolderName := FolderName;
    FBaseCount := BaseCount;
    FCurrentVolumeList := VolumeList;
    FThreadList.StartProc(@ResizeImages_NTL);
  end;

  procedure TCifar10Resize.ResizeImages_NTL(index, threadnum: integer);
  var
    ImgCnt: integer;
    CurrentImg: TNNetVolume;
    ClassId: integer;
    StartPos, FinishPos: integer;
    NN32to64: THistoricalNets;
    NN64to128: THistoricalNets;
    NN128to256: THistoricalNets;
    Aux64Vol: TNNetVolume;
    Aux128Vol: TNNetVolume;
    Aux256Vol: TNNetVolume;
  begin
    EnterCriticalSection(FCriticalSection);
    WriteLn('Creating neural networks at thread ',index,'.');
    NN32to64 := CreateResizingNN(32, 32, csExampleFileName);
    NN64to128 := CreateResizingNN(64, 64, csExampleFileName);
    NN128to256 := CreateResizingNN(128, 128, csExampleFileName);
    Aux64Vol := TNNetVolume.Create();
    Aux128Vol := TNNetVolume.Create();
    Aux256Vol := TNNetVolume.Create();
    FThreadList.CalculateWorkingRange(index, threadnum, FCurrentVolumeList.Count,
      StartPos, FinishPos);
    WriteLn('Thread ',index,' has working range from ',StartPos,' to ',FinishPos,'. This thread is now starting.');
    LeaveCriticalSection(FCriticalSection);
    for ImgCnt := StartPos to FinishPos do
    begin
      if ImgCnt mod 100 = 0 then
      begin
        WriteLn('Thread ',index,' is processing image ', FBaseCount + ImgCnt,' at ', FFolderName,'.');
      end;
      CurrentImg := FCurrentVolumeList[ImgCnt];
      ClassId := CurrentImg.Tag;
      NN32to64.Compute(CurrentImg);
      NN32to64.GetOutput(Aux64Vol);
      NN64to128.Compute(Aux64Vol);
      NN64to128.GetOutput(Aux128Vol);
      NN128to256.Compute(Aux128Vol);
      NN128to256.GetOutput(Aux256Vol);
      CurrentImg.Add(2);
      CurrentImg.Mul(64);
      Aux64Vol.Add(2);
      Aux64Vol.Mul(64);
      Aux128Vol.Add(2);
      Aux128Vol.Mul(64);
      Aux256Vol.Add(2);
      Aux256Vol.Mul(64);
      SaveImageFromVolumeIntoFile(CurrentImg,
        'resized/cifar10-32/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux64Vol,
        'resized/cifar10-64/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux128Vol,
        'resized/cifar10-128/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
      SaveImageFromVolumeIntoFile(Aux256Vol,
        'resized/cifar10-256/'+FFolderName+'/class'+IntToStr(ClassId)+'/'+
        'img'+IntToStr(FBaseCount+ImgCnt)+'.png');
    end;
    WriteLn('Thread ',index,' has finished.');
    Aux256Vol.Free;
    Aux128Vol.Free;
    Aux64Vol.Free;
    NN128to256.Free;
    NN64to128.Free;
    NN32to64.Free;
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

    WriteLn('Creating Threads');
    InitCriticalSection(FCriticalSection);
    FThreadList := TNeuralThreadList.Create( NeuralDefaultThreadCount() );

    ProcessVolumeList(ImgTrainingVolumes, 'train', 0);
    ProcessVolumeList(ImgValidationVolumes, 'train', 40000);
    ProcessVolumeList(ImgTestVolumes, 'test', 0);
    DoneCriticalSection(FCriticalSection);
    FThreadList.Free();
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
