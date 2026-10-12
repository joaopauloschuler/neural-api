unit benchmarkform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, ComCtrls,
  StdCtrls, Math, aicpu, aimemory, aigpu, aidisk, aiso, ai_tasks;

type
  { TfrmBenchmark }

  TfrmBenchmark = class(TForm)
    lblCPUTitle: TLabel;
    lblMemoryTitle: TLabel;
    lvProcesses: TListView;
    memHardware: TMemo;
    paintBoxCPU: TPaintBox;
    paintBoxMemory: TPaintBox;
    pgcBenchmark: TPageControl;
    pnlCPU: TPanel;
    pnlMemory: TPanel;
    timerBenchmark: TTimer;
    tsPerformance: TTabSheet;
    tsProcesses: TTabSheet;
    tsHardware: TTabSheet;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure paintBoxCPUPaint(Sender: TObject);
    procedure paintBoxMemoryPaint(Sender: TObject);
    procedure timerBenchmarkTimer(Sender: TObject);
  private
    FCPU: TAICPU;
    FMemory: TAIMemory;
    FOS: TAIOS;
    FTasks: TAITasks;
    FCPUHistory: array[0..59] of Single;
    FMemHistory: array[0..59] of Single;
    FLanguageIdx: Integer;
    procedure RefreshProcesses;
    procedure RefreshHardwarePage;
    procedure DrawLineChart(APaintBox: TPaintBox; const AHistory: array of Single; AColor: TColor; const ATitle, AValueStr: string);
    procedure UpdateUIStrings;
  public
    procedure SetLanguage(ALangIdx: Integer);
  end;

var
  frmBenchmark: TfrmBenchmark;

implementation

{$R *.lfm}

{ TfrmBenchmark }

procedure TfrmBenchmark.FormCreate(Sender: TObject);
var
  I: Integer;
begin
  Self.DoubleBuffered := True;

  FCPU := TAICPU.Create(Self);
  FMemory := TAIMemory.Create(Self);
  FOS := TAIOS.Create(Self);
  FTasks := TAITasks.Create(Self);

  FTasks.SortBy := sbCPU;
  FTasks.SortDescending := True;

  for I := 0 to 59 do
  begin
    FCPUHistory[I] := 0;
    FMemHistory[I] := 0;
  end;

  FLanguageIdx := 1; // Padrão: Português
  UpdateUIStrings;
end;

procedure TfrmBenchmark.FormDestroy(Sender: TObject);
begin
  timerBenchmark.Enabled := False;
end;

procedure TfrmBenchmark.FormShow(Sender: TObject);
begin
  timerBenchmark.Enabled := True;
  RefreshHardwarePage;
  timerBenchmarkTimer(nil);
end;

procedure TfrmBenchmark.SetLanguage(ALangIdx: Integer);
begin
  FLanguageIdx := ALangIdx;
  UpdateUIStrings;
end;

procedure TfrmBenchmark.UpdateUIStrings;
begin
  // Traduções básicas para a tela de Benchmark
  case FLanguageIdx of
    0: { English }
      begin
        Caption := 'System Performance & Benchmark';
        tsPerformance.Caption := 'Performance';
        tsProcesses.Caption := 'Running Processes';
        tsHardware.Caption := 'Hardware Info';
        lblCPUTitle.Caption := 'CPU Utilization (60s)';
        lblMemoryTitle.Caption := 'Memory Utilization (60s)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Name';
        lvProcesses.Columns[2].Caption := 'CPU %';
        lvProcesses.Columns[3].Caption := 'Memory';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'User';
      end;
    1: { Português }
      begin
        Caption := 'Desempenho do Sistema & Benchmark';
        tsPerformance.Caption := 'Desempenho';
        tsProcesses.Caption := 'Processos Rodando';
        tsHardware.Caption := 'Informações de Hardware';
        lblCPUTitle.Caption := 'Utilização da CPU (60s)';
        lblMemoryTitle.Caption := 'Utilização de Memória (60s)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Nome';
        lvProcesses.Columns[2].Caption := 'CPU %';
        lvProcesses.Columns[3].Caption := 'Memória';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'Usuário';
      end;
    2: { Français }
      begin
        Caption := 'Performances du Système & Benchmark';
        tsPerformance.Caption := 'Performances';
        tsProcesses.Caption := 'Processus en Cours';
        tsHardware.Caption := 'Infos Matériel';
        lblCPUTitle.Caption := 'Utilisation du Processeur (60s)';
        lblMemoryTitle.Caption := 'Utilisation Mémoire (60s)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Nom';
        lvProcesses.Columns[2].Caption := 'Processeur %';
        lvProcesses.Columns[3].Caption := 'Mémoire';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'Utilisateur';
      end;
    3: { Deutsch }
      begin
        Caption := 'Systemleistung & Benchmark';
        tsPerformance.Caption := 'Leistung';
        tsProcesses.Caption := 'Laufende Prozesse';
        tsHardware.Caption := 'Hardware-Info';
        lblCPUTitle.Caption := 'Prozessorauslastung (60s)';
        lblMemoryTitle.Caption := 'Speicherauslastung (60s)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Name';
        lvProcesses.Columns[2].Caption := 'CPU %';
        lvProcesses.Columns[3].Caption := 'Speicher';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'Benutzer';
      end;
    4: { Español }
      begin
        Caption := 'Rendimiento del Sistema & Benchmark';
        tsPerformance.Caption := 'Rendimiento';
        tsProcesses.Caption := 'Procesos Activos';
        tsHardware.Caption := 'Información de Hardware';
        lblCPUTitle.Caption := 'Utilización de CPU (60s)';
        lblMemoryTitle.Caption := 'Utilización de Memoria (60s)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Nombre';
        lvProcesses.Columns[2].Caption := 'CPU %';
        lvProcesses.Columns[3].Caption := 'Memoria';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'Usuario';
      end;
    5: { Arabic }
      begin
        Caption := 'أداء النظام والقياس';
        tsPerformance.Caption := 'الأداء';
        tsProcesses.Caption := 'العمليات الجارية';
        tsHardware.Caption := 'معلومات العتاد';
        lblCPUTitle.Caption := 'استخدام المعالج (60 ثانية)';
        lblMemoryTitle.Caption := 'استخدام الذاكرة (60 ثانية)';
        lvProcesses.Columns[0].Caption := 'معرف العملية';
        lvProcesses.Columns[1].Caption := 'الاسم';
        lvProcesses.Columns[2].Caption := 'المعالج %';
        lvProcesses.Columns[3].Caption := 'الذاكرة';
        lvProcesses.Columns[4].Caption := 'معرف الأب';
        lvProcesses.Columns[5].Caption := 'المستخدم';
      end;
    6: { Chinese }
      begin
        Caption := '系统性能与基准测试';
        tsPerformance.Caption := '性能';
        tsProcesses.Caption := '运行中的进程';
        tsHardware.Caption := '硬件信息';
        lblCPUTitle.Caption := 'CPU 使用率 (60秒)';
        lblMemoryTitle.Caption := '内存使用率 (60秒)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := '名称';
        lvProcesses.Columns[2].Caption := 'CPU %';
        lvProcesses.Columns[3].Caption := '内存';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := '用户';
      end;
    7: { Japanese }
      begin
        Caption := 'システムパフォーマンスとベンチマーク';
        tsPerformance.Caption := 'パフォーマンス';
        tsProcesses.Caption := '実行中のプロセス';
        tsHardware.Caption := 'ハードウェア情報';
        lblCPUTitle.Caption := 'CPU使用率 (60秒)';
        lblMemoryTitle.Caption := 'メモリ使用率 (60秒)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'プロセス名';
        lvProcesses.Columns[2].Caption := 'CPU使用率 %';
        lvProcesses.Columns[3].Caption := 'メモリ';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'ユーザー';
      end;
    8: { Russian }
      begin
        Caption := 'Производительность Системы & Бенчмарк';
        tsPerformance.Caption := 'Производительность';
        tsProcesses.Caption := 'Запущенные процессы';
        tsHardware.Caption := 'Информация о железе';
        lblCPUTitle.Caption := 'Загрузка ЦП (60с)';
        lblMemoryTitle.Caption := 'Использование памяти (60с)';
        lvProcesses.Columns[0].Caption := 'PID';
        lvProcesses.Columns[1].Caption := 'Имя';
        lvProcesses.Columns[2].Caption := 'ЦП %';
        lvProcesses.Columns[3].Caption := 'Память';
        lvProcesses.Columns[4].Caption := 'PPID';
        lvProcesses.Columns[5].Caption := 'Пользователь';
      end;
  end;
end;

procedure TfrmBenchmark.DrawLineChart(APaintBox: TPaintBox; const AHistory: array of Single; AColor: TColor; const ATitle, AValueStr: string);
var
  W, H, I: Integer;
  X, Y, PrevX, PrevY: Integer;
  PointsCount: Integer;
  GridY: Integer;
begin
  W := APaintBox.Width;
  H := APaintBox.Height;

  with APaintBox.Canvas do
  begin
    // Fundo escuro premium
    Brush.Color := RGBToColor(25, 25, 25);
    FillRect(0, 0, W, H);

    // Linhas de Grade (25%, 50%, 75%)
    Pen.Color := RGBToColor(50, 50, 50);
    Pen.Style := psDash;
    for I := 1 to 3 do
    begin
      GridY := Trunc(H * (I / 4));
      MoveTo(0, GridY);
      LineTo(W, GridY);
    end;
    Pen.Style := psSolid;

    // Desenha o gráfico de linha
    Pen.Color := AColor;
    Pen.Width := 2;

    PointsCount := Length(AHistory);
    PrevX := 0;
    PrevY := H - Trunc((AHistory[0] / 100) * H);

    for I := 1 to PointsCount - 1 do
    begin
      X := Trunc((I / (PointsCount - 1)) * W);
      Y := H - Trunc((AHistory[I] / 100) * H);

      // Limita Y para caber na tela
      if Y < 2 then Y := 2;
      if Y > H - 2 then Y := H - 2;

      MoveTo(PrevX, PrevY);
      LineTo(X, Y);

      PrevX := X;
      PrevY := Y;
    end;

    // Texto de Título
    Font.Color := clWhite;
    Font.Size := 9;
    Font.Style := [];
    Brush.Style := bsClear;
    TextOut(10, 10, ATitle);

    // Valor Atual
    Font.Color := AColor;
    Font.Style := [fsBold];
    TextOut(W - TextWidth(AValueStr) - 10, 10, AValueStr);
    Font.Style := [];
  end;
end;

procedure TfrmBenchmark.paintBoxCPUPaint(Sender: TObject);
begin
  DrawLineChart(paintBoxCPU, FCPUHistory, clLime, 'CPU', FormatFloat('0.0%', FCPUHistory[59]));
end;

procedure TfrmBenchmark.paintBoxMemoryPaint(Sender: TObject);
var
  UsedMB, TotalMB: Integer;
begin
  UsedMB := FMemory.LastInfo.UsedMB;
  TotalMB := FMemory.LastInfo.TotalMB;
  DrawLineChart(paintBoxMemory, FMemHistory, clAqua, 'RAM', Format('%d MB / %d MB (%.1f%%)', [UsedMB, TotalMB, FMemHistory[59]]));
end;

procedure TfrmBenchmark.timerBenchmarkTimer(Sender: TObject);
var
  CpuInfo: TAICPUInfo;
  MemInfo: TAIMemoryInfo;
begin
  // Evita reentrância
  timerBenchmark.Enabled := False;
  try
    CpuInfo := FCPU.RefreshInfo;
    MemInfo := FMemory.RefreshInfo;

    // Desloca histórico de CPU
    Move(FCPUHistory[1], FCPUHistory[0], SizeOf(Single) * 59);
    FCPUHistory[59] := CpuInfo.UsageTotalPercent;

    // Desloca histórico de Memória
    Move(FMemHistory[1], FMemHistory[0], SizeOf(Single) * 59);
    FMemHistory[59] := MemInfo.LoadPercent;

    // Redesenha gráficos
    paintBoxCPU.Invalidate;
    paintBoxMemory.Invalidate;

    // Atualiza processos se a aba correspondente estiver ativa
    if pgcBenchmark.ActivePage = tsProcesses then
      RefreshProcesses;
  finally
    timerBenchmark.Enabled := True;
  end;
end;

procedure TfrmBenchmark.RefreshProcesses;
var
  I, J: Integer;
  Item: TListItem;
  T: TAITask;
  UsedMB: Double;
  PIDStr: string;
  Found: Boolean;
  ValPPID, ValName, ValCPU, ValMem, ValUser: string;

  procedure SetSubItem(AItem: TListItem; AIndex: Integer; const AValue: string);
  begin
    while AItem.SubItems.Count <= AIndex do
      AItem.SubItems.Add('');
    if AItem.SubItems[AIndex] <> AValue then
      AItem.SubItems[AIndex] := AValue;
  end;

begin
  FTasks.Refresh;
  lvProcesses.Items.BeginUpdate;
  try
    for I := 0 to lvProcesses.Items.Count - 1 do
      lvProcesses.Items[I].Data := nil;

    for I := 0 to FTasks.Count - 1 do
    begin
      T := FTasks.Tasks[I];
      PIDStr := IntToStr(T.PID);
      Found := False;
      Item := nil;
      for J := 0 to lvProcesses.Items.Count - 1 do
      begin
        if lvProcesses.Items[J].Caption = PIDStr then
        begin
          Item := lvProcesses.Items[J];
          Found := True;
          Break;
        end;
      end;
      if not Found then
      begin
        Item := lvProcesses.Items.Add;
        Item.Caption := PIDStr;
      end;
      ValPPID := IntToStr(T.PPID);
      ValName := T.Name;
      ValCPU := FormatFloat('0.0', T.CPUPercent);
      UsedMB := T.MemoryWorking / 1024 / 1024;
      ValMem := FormatFloat('0.0 MB', UsedMB);
      ValUser := T.User;

      SetSubItem(Item, 0, ValName);
      SetSubItem(Item, 1, ValCPU);
      SetSubItem(Item, 2, ValMem);
      SetSubItem(Item, 3, ValPPID);
      SetSubItem(Item, 4, ValUser);
      Item.Data := Pointer(1);
    end;

    for I := lvProcesses.Items.Count - 1 downto 0 do
    begin
      Item := lvProcesses.Items[I];
      if Item.Data = nil then
        Item.Delete
      else
        Item.Data := nil;
    end;
  finally
    lvProcesses.Items.EndUpdate;
  end;
end;

procedure TfrmBenchmark.RefreshHardwarePage;
var
  CpuInfo: TAICPUInfo;
  MemInfo: TAIMemoryInfo;
  OSInfo: TAIOSInfo;
begin
  CpuInfo := FCPU.RefreshInfo;
  MemInfo := FMemory.RefreshInfo;
  OSInfo := FOS.RefreshInfo;

  memHardware.Lines.BeginUpdate;
  try
    memHardware.Clear;
    memHardware.Lines.Add('=== PROCESSADOR (CPU) ===');
    memHardware.Lines.Add(Format('Processador ID: %s', [CpuInfo.ProcessorId]));
    memHardware.Lines.Add(Format('Núcleos Físicos: %d', [CpuInfo.Cores]));
    memHardware.Lines.Add(Format('Processadores Lógicos: %d', [CpuInfo.LogicalCount]));
    memHardware.Lines.Add(Format('Frequência: %d MHz', [CpuInfo.FrequencyMHz]));
    memHardware.Lines.Add(Format('Cache Line Size: %d bytes', [CpuInfo.CacheLineSize]));
    memHardware.Lines.Add('');
    memHardware.Lines.Add('=== MEMÓRIA RAM ===');
    memHardware.Lines.Add(Format('Total RAM: %d MB', [MemInfo.TotalMB]));
    memHardware.Lines.Add(Format('Disponível RAM: %d MB', [MemInfo.AvailableMB]));
    memHardware.Lines.Add(Format('RAM em Uso: %d MB (%.1f%%)', [MemInfo.UsedMB, MemInfo.LoadPercent]));
    memHardware.Lines.Add(Format('Slots/Pentes: %d', [MemInfo.SlotCount]));
    memHardware.Lines.Add('');
    memHardware.Lines.Add('=== SISTEMA OPERACIONAL ===');
    memHardware.Lines.Add(Format('SO: %s', [OSInfo.OSName]));
    memHardware.Lines.Add(Format('Versão: %s', [OSInfo.OSVersion]));
    memHardware.Lines.Add(Format('Arquitetura: %s (%s bits)', [OSInfo.Architecture, OSInfo.Bitness]));
  finally
    memHardware.Lines.EndUpdate;
  end;
end;

end.
