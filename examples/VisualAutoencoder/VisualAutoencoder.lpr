program VisualAutoencoder;

{$mode objfpc}{$H+}

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, uvisualautoencodertinyimagenet, neuralfit;

{$R *.res}

begin
  RequireDerivedFormResource:=True;
  Application.Initialize;
  Application.CreateForm(TFormVisualLearning, FormVisualLearning);
  Application.Run;
end.

