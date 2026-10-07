unit mainform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, Menus,
  StdCtrls, ComCtrls, Process, IniFiles, fphttpclient, opensslsockets, Types, Registry;

type
  TTranslation = record
    StatusLabelOffline: string;
    StatusLabelOnline: string;
    StartMenu: string;
    StopMenu: string;
    LogMenu: string;
    LanguageMenu: string;
    ExitMenu: string;
    FormTitle: string;
    ConfigTitle: string;
    PortLabel: string;
    ModelLabel: string;
    PathLabel: string;
    ServerPathLabel: string;
    ModelsPathLabel: string;
    TabConfig: string;
    TabLog: string;
    BtnClear: string;
    BtnSave: string;
    BtnCancel: string;
    BtnBenchmark: string;
    BtnChat: string;
    CtxLabel: string;
    MaxNewTokensLabel: string;
    AutoStartLabel: string;
    AutoOnLabel: string;
    GpuLabel: string;
    BtnSetup: string;
  end;

  { TfrmManager }

  TfrmManager = class(TForm)
    btnBrowsePath: TButton;
    btnBrowseServer: TButton;
    btnBrowseModels: TButton;
    btnSave: TButton;
    btnCancelForm: TButton;
    btnClearLog: TButton;
    btnBenchmark: TButton;
    btnChat: TButton;
    btnSetup: TButton;
    cmbModel: TComboBox;
    cmbLanguage: TComboBox;
    cmbMaxNewTokens: TComboBox;
    chkAutoStart: TCheckBox;
    chkAutoOn: TCheckBox;
    chkUseGpu: TCheckBox;
    edtProjectPath: TEdit;
    edtServerPath: TEdit;
    edtModelsPath: TEdit;
    edtPort: TEdit;
    edtCtx: TEdit;
    lblProjectPath: TLabel;
    lblServerPath: TLabel;
    lblModelsPath: TLabel;
    lblPort: TLabel;
    lblModel: TLabel;
    lblLanguage: TLabel;
    lblCtx: TLabel;
    lblMaxNewTokens: TLabel;
    memLog: TMemo;
    pgcMain: TPageControl;
    tsConfig: TTabSheet;
    tsLog: TTabSheet;
    selectDirDialog: TSelectDirectoryDialog;
    timerStatus: TTimer;
    trayIcon: TTrayIcon;
    pmTrayMenu: TPopupMenu;
    menuStatus: TMenuItem;
    menuSeparator1: TMenuItem;
    menuStart: TMenuItem;
    menuStop: TMenuItem;
    menuSeparator4: TMenuItem;
    menuChat: TMenuItem;
    menuBenchmark: TMenuItem;
    menuSeparator2: TMenuItem;
    menuConfig: TMenuItem;
    menuLog: TMenuItem;
    menuLanguage: TMenuItem;
    menuSeparator3: TMenuItem;
    menuExit: TMenuItem;
    procedure btnBrowsePathClick(Sender: TObject);
    procedure btnBrowseServerClick(Sender: TObject);
    procedure btnBrowseModelsClick(Sender: TObject);
    procedure btnSaveClick(Sender: TObject);
    procedure btnCancelFormClick(Sender: TObject);
    procedure btnClearLogClick(Sender: TObject);
    procedure btnBenchmarkClick(Sender: TObject);
    procedure btnChatClick(Sender: TObject);
    procedure btnSetupClick(Sender: TObject);
    procedure cmbLanguageChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure menuStartClick(Sender: TObject);
    procedure menuStopClick(Sender: TObject);
    procedure menuChatClick(Sender: TObject);
    procedure menuBenchmarkClick(Sender: TObject);
    procedure menuConfigClick(Sender: TObject);
    procedure menuLogClick(Sender: TObject);
    procedure menuExitClick(Sender: TObject);
    procedure timerStatusTimer(Sender: TObject);
    procedure trayIconDblClick(Sender: TObject);
  private
    FProjectPath: string;
    FChatServerPath: string;
    FModelsPath: string;
    FPort: string;
    FCtx: string;
    FMaxNewTokens: string;
    FAutoStart: Boolean;
    FAutoOn: Boolean;
    FUseGpu: Boolean;
    FLastStartAttempt: TDateTime;
    FSelectedModel: string;
    FLanguageIdx: Integer;
    FServerActive: Boolean;
    FServerProcess: TProcess;
    FActiveIcon: TIcon;
    FInactiveIcon: TIcon;
    procedure SetAutoStartWindows(const AEnable: Boolean);
    function GenerateTrayIcon(AActive: Boolean): TIcon;
    function GetConfigPath: string;
    procedure LoadConfig;
    procedure SaveConfig;
    procedure UpdateModelsCombo;
    procedure UpdateUIStrings;
    procedure menuLanguageItemClick(Sender: TObject);
    function IsServerOnline: Boolean;
    procedure StartServer;
    procedure StopServer;
    procedure KillAllChatServers;
    procedure UpdateStatusUI;
    procedure LogEvent(const AMsg: string);
  public
  end;

var
  frmManager: TfrmManager;

const
  Translations: array[0..8] of TTranslation = (
    // 0: English
    (
      StatusLabelOffline: 'ChatServer: Offline';
      StatusLabelOnline: 'ChatServer: Online';
      StartMenu: 'Start ChatServer';
      StopMenu: 'Stop ChatServer';
      LogMenu: 'Show Log';
      LanguageMenu: 'Language';
      ExitMenu: 'Exit';
      FormTitle: 'ChatServer Manager';
      ConfigTitle: 'Configuration';
      PortLabel: 'Port:';
      ModelLabel: 'Model Folder:';
      PathLabel: 'Project Path:';
      ServerPathLabel: 'ChatServer.exe Path:';
      ModelsPathLabel: 'Models Folder Path:';
      TabConfig: 'Configuration';
      TabLog: 'Log Viewer';
      BtnClear: 'Clear';
      BtnSave: 'Save';
      BtnCancel: 'Cancel';
      BtnBenchmark: 'Benchmark';
      BtnChat: 'Chat';
      CtxLabel: 'Context (ctx):';
      MaxNewTokensLabel: 'Max New Tokens:';
      AutoStartLabel: 'Auto Start (Windows Startup)';
      AutoOnLabel: 'Auto On (Auto Restart Server)';
      GpuLabel: 'Use GPU (OpenCL)';
      BtnSetup: 'Setup'
    ),
    // 1: Português
    (
      StatusLabelOffline: 'ChatServer: Offline';
      StatusLabelOnline: 'ChatServer: Online';
      StartMenu: 'Iniciar ChatServer';
      StopMenu: 'Parar ChatServer';
      LogMenu: 'Exibir Log';
      LanguageMenu: 'Idioma';
      ExitMenu: 'Sair';
      FormTitle: 'Gerenciador do ChatServer';
      ConfigTitle: 'Configuração';
      PortLabel: 'Porta:';
      ModelLabel: 'Pasta do Modelo:';
      PathLabel: 'Caminho do Projeto:';
      ServerPathLabel: 'Caminho do ChatServer.exe:';
      ModelsPathLabel: 'Caminho da Pasta de Modelos:';
      TabConfig: 'Configuração';
      TabLog: 'Visualizador de Log';
      BtnClear: 'Limpar';
      BtnSave: 'Salvar';
      BtnCancel: 'Cancelar';
      BtnBenchmark: 'Benchmark';
      BtnChat: 'Chat';
      CtxLabel: 'Contexto (ctx):';
      MaxNewTokensLabel: 'Max Novos Tokens:';
      AutoStartLabel: 'Auto Start (Iniciar com o Windows)';
      AutoOnLabel: 'Auto On (Auto reiniciar ChatServer)';
      GpuLabel: 'Usar GPU (OpenCL)';
      BtnSetup: 'Setup'
    ),
    // 2: Français
    (
      StatusLabelOffline: 'ChatServer : Hors ligne';
      StatusLabelOnline: 'ChatServer : En ligne';
      StartMenu: 'Démarrer le ChatServer';
      StopMenu: 'Arrêter le ChatServer';
      LogMenu: 'Afficher le Log';
      LanguageMenu: 'Langue';
      ExitMenu: 'Quitter';
      FormTitle: 'Gestionnaire ChatServer';
      ConfigTitle: 'Configuration';
      PortLabel: 'Port :';
      ModelLabel: 'Dossier du Modèle :';
      PathLabel: 'Chemin du Projet :';
      ServerPathLabel: 'Chemin du ChatServer.exe :';
      ModelsPathLabel: 'Chemin du dossier Modèles :';
      TabConfig: 'Configuration';
      TabLog: 'Visionneuse de Log';
      BtnClear: 'Effacer';
      BtnSave: 'Enregistrer';
      BtnCancel: 'Annuler';
      BtnBenchmark: 'Benchmark';
      BtnChat: 'Chat';
      CtxLabel: 'Contexte (ctx) :';
      MaxNewTokensLabel: 'Max Nouveaux Tokens :';
      AutoStartLabel: 'Auto Start (Démarrer avec Windows)';
      AutoOnLabel: 'Auto On (Redémarrer serveur auto)';
      GpuLabel: 'Utiliser GPU (OpenCL)';
      BtnSetup: 'Setup'
    ),
    // 3: Deutsch
    (
      StatusLabelOffline: 'ChatServer: Offline';
      StatusLabelOnline: 'ChatServer: Online';
      StartMenu: 'ChatServer starten';
      StopMenu: 'ChatServer stoppen';
      LogMenu: 'Protokoll anzeigen';
      LanguageMenu: 'Sprache';
      ExitMenu: 'Beenden';
      FormTitle: 'ChatServer Manager';
      ConfigTitle: 'Konfiguration';
      PortLabel: 'Port:';
      ModelLabel: 'Modellordner:';
      PathLabel: 'Projektpfad:';
      ServerPathLabel: 'Pfad zu ChatServer.exe:';
      ModelsPathLabel: 'Pfad zum Modellordner:';
      TabConfig: 'Konfiguration';
      TabLog: 'Protokollanzeige';
      BtnClear: 'Löschen';
      BtnSave: 'Speichern';
      BtnCancel: 'Abbrechen';
      BtnBenchmark: 'Benchmark';
      BtnChat: 'Chat';
      CtxLabel: 'Kontext (ctx):';
      MaxNewTokensLabel: 'Max Neue Token:';
      AutoStartLabel: 'Auto Start (Mit Windows starten)';
      AutoOnLabel: 'Auto On (Server auto neustarten)';
      GpuLabel: 'GPU (OpenCL) verwenden';
      BtnSetup: 'Setup'
    ),
    // 4: Español
    (
      StatusLabelOffline: 'ChatServer: Desconectado';
      StatusLabelOnline: 'ChatServer: En línea';
      StartMenu: 'Iniciar ChatServer';
      StopMenu: 'Detener ChatServer';
      LogMenu: 'Mostrar registro';
      LanguageMenu: 'Idioma';
      ExitMenu: 'Salir';
      FormTitle: 'Administrador de ChatServer';
      ConfigTitle: 'Configuración';
      PortLabel: 'Puerto:';
      ModelLabel: 'Carpeta del modelo:';
      PathLabel: 'Ruta del proyecto:';
      ServerPathLabel: 'Ruta de ChatServer.exe:';
      ModelsPathLabel: 'Ruta de la carpeta de modelos:';
      TabConfig: 'Configuración';
      TabLog: 'Visor de registro';
      BtnClear: 'Limpiar';
      BtnSave: 'Guardar';
      BtnCancel: 'Cancelar';
      BtnBenchmark: 'Benchmark';
      BtnChat: 'Chat';
      CtxLabel: 'Contexto (ctx):';
      MaxNewTokensLabel: 'Máx Nuevos Tokens:';
      AutoStartLabel: 'Auto Start (Iniciar con Windows)';
      AutoOnLabel: 'Auto On (Reiniciar servidor auto)';
      GpuLabel: 'Usar GPU (OpenCL)';
      BtnSetup: 'Setup'
    ),
    // 5: Arabic
    (
      StatusLabelOffline: 'ChatServer: غير متصل';
      StatusLabelOnline: 'ChatServer: متصل';
      StartMenu: 'تشغيل ChatServer';
      StopMenu: 'إيقاف ChatServer';
      LogMenu: 'عرض السجل';
      LanguageMenu: 'اللغة';
      ExitMenu: 'خروج';
      FormTitle: 'مدير ChatServer';
      ConfigTitle: 'الإعدادات';
      PortLabel: 'المنفذ:';
      ModelLabel: 'مجلد النموذج:';
      PathLabel: 'مسار المشروع:';
      ServerPathLabel: 'مسار ChatServer.exe:';
      ModelsPathLabel: 'مسار مجلد النماذج:';
      TabConfig: 'الإعدادات';
      TabLog: 'عارض السجل';
      BtnClear: 'مسح';
      BtnSave: 'حفظ';
      BtnCancel: 'إلغاء';
      BtnBenchmark: 'اختبار الأداء';
      BtnChat: 'دردشة';
      CtxLabel: 'السياق (ctx):';
      MaxNewTokensLabel: 'الحد الأقصى للرموز:';
      AutoStartLabel: 'بدء تلقائي (مع ويندوز)';
      AutoOnLabel: 'تشغيل تلقائي (إعادة تشغيل الخادم)';
      GpuLabel: 'استخدام وحدة معالجة الرسومات';
      BtnSetup: 'إعداد'
    ),
    // 6: Chinese
    (
      StatusLabelOffline: 'ChatServer: 离线';
      StatusLabelOnline: 'ChatServer: 在线';
      StartMenu: '启动 ChatServer';
      StopMenu: '停止 ChatServer';
      LogMenu: '显示日志';
      LanguageMenu: '语言';
      ExitMenu: '退出';
      FormTitle: 'ChatServer 管理器';
      ConfigTitle: '配置';
      PortLabel: '端口:';
      ModelLabel: '模型文件夹:';
      PathLabel: '项目路径:';
      ServerPathLabel: 'ChatServer.exe 路径:';
      ModelsPathLabel: '模型文件夹路径:';
      TabConfig: '配置';
      TabLog: '日志查看器';
      BtnClear: '清除';
      BtnSave: '保存';
      BtnCancel: '取消';
      BtnBenchmark: '基准测试';
      BtnChat: '聊天';
      CtxLabel: '上下文 (ctx):';
      MaxNewTokensLabel: '最大新 Token:';
      AutoStartLabel: '开机自启 (随 Windows 启动)';
      AutoOnLabel: '自动开机 (自动重启 ChatServer)';
      GpuLabel: '使用 GPU (OpenCL)';
      BtnSetup: '设置'
    ),
    // 7: Japanese
    (
      StatusLabelOffline: 'ChatServer: オフライン';
      StatusLabelOnline: 'ChatServer: オンライン';
      StartMenu: 'ChatServer を起動';
      StopMenu: 'ChatServer を停止';
      LogMenu: 'ログを表示';
      LanguageMenu: '言語';
      ExitMenu: '終了';
      FormTitle: 'ChatServer マネージャー';
      ConfigTitle: '設定';
      PortLabel: 'ポート:';
      ModelLabel: 'モデルフォルダ:';
      PathLabel: 'プロジェクトパス:';
      ServerPathLabel: 'ChatServer.exe パス:';
      ModelsPathLabel: 'モデルフォルダパス:';
      TabConfig: '設定';
      TabLog: 'ログビューア';
      BtnClear: 'クリア';
      BtnSave: '保存';
      BtnCancel: 'キャンセル';
      BtnBenchmark: 'ベンチマーク';
      BtnChat: 'チャット';
      CtxLabel: 'コンテキスト (ctx):';
      MaxNewTokensLabel: '最大新規トークン:';
      AutoStartLabel: '自動起動 (Windows 起動時)';
      AutoOnLabel: '自動 On (サーバー自動再起動)';
      GpuLabel: 'GPU (OpenCL) を使用';
      BtnSetup: 'セットアップ'
    ),
    // 8: Russian
    (
      StatusLabelOffline: 'ChatServer: Офлайн';
      StatusLabelOnline: 'ChatServer: Онлайн';
      StartMenu: 'Запустить ChatServer';
      StopMenu: 'Остановить ChatServer';
      LogMenu: 'Показать лог';
      LanguageMenu: 'Язык';
      ExitMenu: 'Выход';
      FormTitle: 'Менеджер ChatServer';
      ConfigTitle: 'Конфигурация';
      PortLabel: 'Порт:';
      ModelLabel: 'Папка модели:';
      PathLabel: 'Путь к проекту:';
      ServerPathLabel: 'Путь к ChatServer.exe:';
      ModelsPathLabel: 'Путь к папке моделей:';
      TabConfig: 'Конфигурация';
      TabLog: 'Просмотр лога';
      BtnClear: 'Очистить';
      BtnSave: 'Сохранить';
      BtnCancel: 'Отмена';
      BtnBenchmark: 'Бенчмарк';
      BtnChat: 'Чат';
      CtxLabel: 'Контекст (ctx):';
      MaxNewTokensLabel: 'Макс. новых токенов:';
      AutoStartLabel: 'Автозапуск (с Windows)';
      AutoOnLabel: 'Авто On (Автоперезапуск)';
      GpuLabel: 'Использовать GPU (OpenCL)';
      BtnSetup: 'Установка'
    )
  );

implementation

uses benchmarkform, chatform;

{$R *.lfm}

{ TfrmManager }

procedure TfrmManager.SetAutoStartWindows(const AEnable: Boolean);
var
  Reg: TRegistry;
begin
  try
    Reg := TRegistry.Create(KEY_ALL_ACCESS);
    try
      Reg.RootKey := HKEY_CURRENT_USER;
      if Reg.OpenKey('Software\Microsoft\Windows\CurrentVersion\Run', True) then
      begin
        if AEnable then
          Reg.WriteString('ChatServerManager', '"' + Application.ExeName + '"')
        else if Reg.ValueExists('ChatServerManager') then
          Reg.DeleteValue('ChatServerManager');
        Reg.CloseKey;
      end;
    finally
      Reg.Free;
    end;
  except
    on E: Exception do
      LogEvent('Erro ao atualizar Registro do Windows para Auto Start: ' + E.Message);
  end;
end;

procedure TfrmManager.LogEvent(const AMsg: string);
var
  TimestampedMsg, LogDir, LogFilePath: string;
  F: TextFile;
begin
  TimestampedMsg := FormatDateTime('yyyy-mm-dd hh:nn:ss', Now) + ' - ' + AMsg;
  memLog.Lines.Add(TimestampedMsg);

  LogDir := IncludeTrailingPathDelimiter(FProjectPath) + 'manager' + PathDelim + 'log';
  try
    ForceDirectories(LogDir);
    LogFilePath := IncludeTrailingPathDelimiter(LogDir) + 'manager.log';
    AssignFile(F, LogFilePath);
    if FileExists(LogFilePath) then
      Append(F)
    else
      Rewrite(F);
    WriteLn(F, TimestampedMsg);
    CloseFile(F);
  except
    // Silent catch so logging failure does not interrupt UI operations
  end;
end;

function TfrmManager.GetConfigPath: string;
var
  LocalPath, AppDir: string;
begin
  LocalPath := ExtractFilePath(Application.ExeName) + 'manager.ini';
  if FileExists(LocalPath) then
    Exit(LocalPath);

  AppDir := GetAppConfigDir(False);
  if not DirectoryExists(AppDir) then
    ForceDirectories(AppDir);
  Result := IncludeTrailingPathDelimiter(AppDir) + 'manager.ini';
end;

function TfrmManager.GenerateTrayIcon(AActive: Boolean): TIcon;
var
  Bmp: TBitmap;
  ImgList: TImageList;
  C: TCanvas;
  I, J: Integer;
  InPoints, HidPoints, OutPoints: array of TPoint;
  NodeColor, LineColor: TColor;
begin
  Result := TIcon.Create;
  Bmp := TBitmap.Create;
  try
    Bmp.Width := 32;
    Bmp.Height := 32;
    C := Bmp.Canvas;

    C.Brush.Color := clFuchsia;
    C.FillRect(0, 0, 32, 32);

    SetLength(InPoints, 3);
    InPoints[0] := Point(6, 6);
    InPoints[1] := Point(6, 16);
    InPoints[2] := Point(6, 26);

    SetLength(HidPoints, 4);
    HidPoints[0] := Point(16, 4);
    HidPoints[1] := Point(16, 12);
    HidPoints[2] := Point(16, 20);
    HidPoints[3] := Point(16, 28);

    SetLength(OutPoints, 2);
    OutPoints[0] := Point(26, 10);
    OutPoints[1] := Point(26, 22);

    if AActive then
      LineColor := RGBToColor(100, 180, 255)
    else
      LineColor := clDkGray;

    C.Pen.Color := LineColor;
    C.Pen.Width := 1;

    for I := 0 to 2 do
      for J := 0 to 3 do
      begin
        C.MoveTo(InPoints[I].X, InPoints[I].Y);
        C.LineTo(HidPoints[J].X, HidPoints[J].Y);
      end;

    for I := 0 to 3 do
      for J := 0 to 1 do
      begin
        C.MoveTo(HidPoints[I].X, HidPoints[I].Y);
        C.LineTo(OutPoints[J].X, OutPoints[J].Y);
      end;

    for I := 0 to 2 do
    begin
      if AActive then NodeColor := clRed else NodeColor := clGray;
      C.Brush.Color := NodeColor;
      if AActive then C.Pen.Color := clWhite else C.Pen.Color := clDkGray;
      C.Ellipse(InPoints[I].X - 3, InPoints[I].Y - 3, InPoints[I].X + 3, InPoints[I].Y + 3);
    end;

    for I := 0 to 3 do
    begin
      if AActive then NodeColor := clLime else NodeColor := clGray;
      C.Brush.Color := NodeColor;
      if AActive then C.Pen.Color := clWhite else C.Pen.Color := clDkGray;
      C.Ellipse(HidPoints[I].X - 3, HidPoints[I].Y - 3, HidPoints[I].X + 3, HidPoints[I].Y + 3);
    end;

    for I := 0 to 1 do
    begin
      if AActive then NodeColor := clYellow else NodeColor := clGray;
      C.Brush.Color := NodeColor;
      if AActive then C.Pen.Color := clWhite else C.Pen.Color := clDkGray;
      C.Ellipse(OutPoints[I].X - 3, OutPoints[I].Y - 3, OutPoints[I].X + 3, OutPoints[I].Y + 3);
    end;

    ImgList := TImageList.CreateSize(32, 32);
    try
      ImgList.AddMasked(Bmp, clFuchsia);
      ImgList.GetIcon(0, Result);
    finally
      ImgList.Free;
    end;

  finally
    Bmp.Free;
  end;
end;

procedure TfrmManager.LoadConfig;
var
  Ini: TIniFile;
begin
  Ini := TIniFile.Create(GetConfigPath);
  try
    FProjectPath := Ini.ReadString('Settings', 'ProjectPath', '');
    if FProjectPath = '' then
      FProjectPath := ExtractFilePath(ExcludeTrailingPathDelimiter(ExtractFileDir(Application.ExeName)));

    if not FileExists(IncludeTrailingPathDelimiter(FProjectPath) + 'examples' + PathDelim + 'ChatTerminal' + PathDelim + 'ChatServer.lpi') then
      FProjectPath := 'D:\projetos\neural-api';

    FChatServerPath := Ini.ReadString('Settings', 'ChatServerPath', '');
    FModelsPath := Ini.ReadString('Settings', 'ModelsPath', '');

    FPort := Ini.ReadString('Settings', 'Port', '8095');
    FCtx := Ini.ReadString('Settings', 'Ctx', '8192');
    FMaxNewTokens := Ini.ReadString('Settings', 'MaxNewTokens', '32');
    FAutoStart := Ini.ReadBool('Settings', 'AutoStart', False);
    FAutoOn := Ini.ReadBool('Settings', 'AutoOn', False);
    FUseGpu := Ini.ReadBool('Settings', 'UseGpu', False);
    FSelectedModel := Ini.ReadString('Settings', 'Model', 'Qwen2.5-0.5B-Instruct');
    FLanguageIdx := Ini.ReadInteger('Settings', 'Language', 1);
  finally
    Ini.Free;
  end;
end;

procedure TfrmManager.SaveConfig;
var
  Ini: TIniFile;
begin
  Ini := TIniFile.Create(GetConfigPath);
  try
    Ini.WriteString('Settings', 'ProjectPath', FProjectPath);
    Ini.WriteString('Settings', 'ChatServerPath', FChatServerPath);
    Ini.WriteString('Settings', 'ModelsPath', FModelsPath);
    Ini.WriteString('Settings', 'Port', FPort);
    Ini.WriteString('Settings', 'Ctx', FCtx);
    Ini.WriteString('Settings', 'MaxNewTokens', FMaxNewTokens);
    Ini.WriteBool('Settings', 'AutoStart', FAutoStart);
    Ini.WriteBool('Settings', 'AutoOn', FAutoOn);
    Ini.WriteBool('Settings', 'UseGpu', FUseGpu);
    Ini.WriteString('Settings', 'Model', FSelectedModel);
    Ini.WriteInteger('Settings', 'Language', FLanguageIdx);
  finally
    Ini.Free;
  end;

  SetAutoStartWindows(FAutoStart);
  LogEvent('Configurações salvas.');
end;

procedure TfrmManager.UpdateModelsCombo;
var
  SR: TSearchRec;
  ModelsDir: string;
begin
  cmbModel.Items.Clear;

  if FModelsPath <> '' then
    ModelsDir := FModelsPath
  else
    ModelsDir := IncludeTrailingPathDelimiter(FProjectPath) + 'models';

  if DirectoryExists(ModelsDir) then
  begin
    if FindFirst(ModelsDir + PathDelim + '*', faDirectory, SR) = 0 then
    begin
      repeat
        if ((SR.Attr and faDirectory) <> 0) and (SR.Name <> '.') and (SR.Name <> '..') then
        begin
          cmbModel.Items.Add(SR.Name);
        end;
      until FindNext(SR) <> 0;
      FindClose(SR);
    end;
  end;

  if cmbModel.Items.Count = 0 then
    cmbModel.Items.Add('Qwen2.5-0.5B-Instruct');

  if cmbModel.Items.IndexOf(FSelectedModel) >= 0 then
    cmbModel.ItemIndex := cmbModel.Items.IndexOf(FSelectedModel)
  else
    cmbModel.ItemIndex := 0;
end;

procedure TfrmManager.UpdateUIStrings;
var
  Trans: TTranslation;
  I: Integer;
  MenuItem: TMenuItem;
begin
  Trans := Translations[FLanguageIdx];

  Caption := Trans.FormTitle;
  lblProjectPath.Caption := Trans.PathLabel;
  lblServerPath.Caption := Trans.ServerPathLabel;
  lblModelsPath.Caption := Trans.ModelsPathLabel;
  lblPort.Caption := Trans.PortLabel;
  lblModel.Caption := Trans.ModelLabel;
  lblLanguage.Caption := Trans.LanguageMenu + ':';
  lblCtx.Caption := Trans.CtxLabel;
  lblMaxNewTokens.Caption := Trans.MaxNewTokensLabel;
  chkAutoStart.Caption := Trans.AutoStartLabel;
  chkAutoOn.Caption := Trans.AutoOnLabel;
  chkUseGpu.Caption := Trans.GpuLabel;
  btnSetup.Caption := Trans.BtnSetup;
  btnSave.Caption := Trans.BtnSave;
  btnCancelForm.Caption := Trans.BtnCancel;
  btnClearLog.Caption := Trans.BtnClear;
  btnBenchmark.Caption := Trans.BtnBenchmark;
  btnChat.Caption := Trans.BtnChat;

  tsConfig.Caption := Trans.TabConfig;
  tsLog.Caption := Trans.TabLog;

  if FServerActive then
    menuStatus.Caption := Trans.StatusLabelOnline
  else
    menuStatus.Caption := Trans.StatusLabelOffline;

  menuStart.Caption := Trans.StartMenu;
  menuStop.Caption := Trans.StopMenu;
  menuChat.Caption := Trans.BtnChat;
  menuBenchmark.Caption := Trans.BtnBenchmark;
  menuConfig.Caption := Trans.TabConfig;
  menuLog.Caption := Trans.LogMenu;
  menuLanguage.Caption := Trans.LanguageMenu;
  menuExit.Caption := Trans.ExitMenu;

  menuLanguage.Clear;
  for I := 0 to 8 do
  begin
    MenuItem := TMenuItem.Create(menuLanguage);
    case I of
      0: MenuItem.Caption := 'English';
      1: MenuItem.Caption := 'Português';
      2: MenuItem.Caption := 'Français';
      3: MenuItem.Caption := 'Deutsch';
      4: MenuItem.Caption := 'Español';
      5: MenuItem.Caption := 'العربية';
      6: MenuItem.Caption := '中文';
      7: MenuItem.Caption := '日本語';
      8: MenuItem.Caption := 'Русский';
    end;
    MenuItem.Tag := I;
    MenuItem.Checked := (I = FLanguageIdx);
    MenuItem.OnClick := @menuLanguageItemClick;
    menuLanguage.Add(MenuItem);
  end;

  if FServerActive then
    trayIcon.Hint := Trans.FormTitle + ' (' + Trans.StatusLabelOnline + ')'
  else
    trayIcon.Hint := Trans.FormTitle + ' (' + Trans.StatusLabelOffline + ')';

  if Assigned(frmBenchmark) then
    frmBenchmark.SetLanguage(FLanguageIdx);

  if Assigned(frmChat) then
    frmChat.SetConfig(edtPort.Text, cmbModel.Text, FLanguageIdx);
end;

procedure TfrmManager.menuLanguageItemClick(Sender: TObject);
begin
  if Sender is TMenuItem then
  begin
    FLanguageIdx := TMenuItem(Sender).Tag;
    cmbLanguage.ItemIndex := FLanguageIdx;
    LogEvent('Idioma alterado no menu para: ' + cmbLanguage.Items[FLanguageIdx]);
    UpdateUIStrings;
    SaveConfig;
  end;
end;

function TfrmManager.IsServerOnline: Boolean;
var
  Client: TFPHttpClient;
  Res: string;
begin
  Result := False;
  Client := TFPHttpClient.Create(nil);
  try
    Client.ConnectTimeout := 1;
    Client.IOTimeout := 1;
    Res := Client.Get('http://127.0.0.1:' + FPort + '/v1/models');
    Result := (Res <> '');
  except
    // Silent fail
  end;
  Client.Free;
end;

procedure TfrmManager.StartServer;
var
  ServerExePath, ModelPath: string;
begin
  if FServerActive then Exit;

  LogEvent('Solicitação de inicialização do ChatServer.');

  // Determina o executável do ChatServer
  if FChatServerPath <> '' then
    ServerExePath := FChatServerPath
  else
  begin
    ServerExePath := IncludeTrailingPathDelimiter(FProjectPath) + 'bin' + PathDelim + 'x86_64-win64' + PathDelim + 'bin' + PathDelim + 'ChatServer.exe';
    if not FileExists(ServerExePath) then
      ServerExePath := IncludeTrailingPathDelimiter(FProjectPath) + 'examples' + PathDelim + 'ChatTerminal' + PathDelim + 'ChatServer.exe';
  end;

  if not FileExists(ServerExePath) then
  begin
    LogEvent('Erro: Executável do ChatServer não encontrado em: ' + ServerExePath);
    ShowMessage('ChatServer.exe não encontrado em ' + ServerExePath + '. Por favor, compile o servidor no instalador ou informe a rota nas configurações.');
    Exit;
  end;

  if FModelsPath <> '' then
    ModelPath := IncludeTrailingPathDelimiter(FModelsPath) + FSelectedModel
  else
    ModelPath := IncludeTrailingPathDelimiter(FProjectPath) + 'models' + PathDelim + FSelectedModel;

  LogEvent('Iniciando binário: ' + ServerExePath);
  LogEvent('Modelo associado: ' + ModelPath);
  LogEvent('Porta selecionada: ' + FPort);
  LogEvent('Contexto (ctx): ' + FCtx);
  LogEvent('Max New Tokens: ' + FMaxNewTokens);
  if FUseGpu then
    LogEvent('>>> HARDWARE SELECIONADO: GPU (OpenCL Aceleração de Hardware) <<<')
  else
    LogEvent('>>> HARDWARE SELECIONADO: CPU (Processador do Host - Sem GPU) <<<');

  FServerProcess := TProcess.Create(nil);
  FServerProcess.Executable := ServerExePath;
  FServerProcess.Parameters.Add(ModelPath);
  FServerProcess.Parameters.Add('--ctx');
  FServerProcess.Parameters.Add(FCtx);
  FServerProcess.Parameters.Add('--max-new-tokens');
  FServerProcess.Parameters.Add(FMaxNewTokens);
  FServerProcess.Parameters.Add('--max-fast-memory');
  FServerProcess.Parameters.Add('--stats');
  FServerProcess.Parameters.Add('--port');
  FServerProcess.Parameters.Add(FPort);
  if FUseGpu then
    FServerProcess.Parameters.Add('--gpu')
  else
  begin
    FServerProcess.Parameters.Add('--cpu');
    FServerProcess.Parameters.Add('--serial');
  end;
  FServerProcess.Parameters.Add('--log-file');
  FServerProcess.Parameters.Add(IncludeTrailingPathDelimiter(FProjectPath) + 'manager' + PathDelim + 'log' + PathDelim + 'chatserver.log');
  FServerProcess.Options := [poNoConsole];
  FServerProcess.ShowWindow := swoNone;

  try
    FServerProcess.Execute;
    LogEvent('Processo executado no SO.');
    Sleep(500);
    timerStatusTimer(nil);
  except
    on E: Exception do
    begin
      LogEvent('Erro fatal ao iniciar ChatServer: ' + E.Message);
      ShowMessage('Erro ao iniciar o servidor: ' + E.Message);
    end;
  end;
end;

procedure TfrmManager.StopServer;
begin
  LogEvent('Solicitação de parada do ChatServer.');
  if Assigned(FServerProcess) then
  begin
    if FServerProcess.Active then
    begin
      FServerProcess.Terminate(0);
      LogEvent('Processo gerenciado finalizado pelo manager.');
    end;
    FreeAndNil(FServerProcess);
  end;
  KillAllChatServers;
  FServerActive := False;
  UpdateStatusUI;
end;

procedure TfrmManager.KillAllChatServers;
var
  Proc: TProcess;
begin
  LogEvent('Matando todas as instâncias de ChatServer.exe executando taskkill...');
  Proc := TProcess.Create(nil);
  try
    Proc.Executable := 'taskkill.exe';
    Proc.Parameters.Add('/f');
    Proc.Parameters.Add('/im');
    Proc.Parameters.Add('ChatServer.exe');
    Proc.Options := [poNoConsole];
    Proc.Execute;
  except
    on E: Exception do
      LogEvent('Taskkill retornou erro (provavelmente nenhum processo estava ativo): ' + E.Message);
  end;
  Proc.Free;
end;

procedure TfrmManager.UpdateStatusUI;
begin
  if FServerActive then
  begin
    trayIcon.Icon := FActiveIcon;
    menuStart.Enabled := False;
    menuStop.Enabled := True;
  end
  else
  begin
    trayIcon.Icon := FInactiveIcon;
    menuStart.Enabled := True;
    menuStop.Enabled := False;
  end;
  UpdateUIStrings;
end;

procedure TfrmManager.FormCreate(Sender: TObject);
begin
  Application.ShowMainForm := False;
  FServerActive := False;
  FServerProcess := nil;

  LogEvent('=== INICIALIZAÇÃO DO CHATSERVER MANAGER ===');
  LogEvent('Local do executável: ' + Application.ExeName);

  LoadConfig;
  LogEvent('Configuração carregada. Idioma: ' + IntToStr(FLanguageIdx));
  LogEvent('Pasta do AppData em uso: ' + ExtractFilePath(GetConfigPath));

  FActiveIcon := GenerateTrayIcon(True);
  FInactiveIcon := GenerateTrayIcon(False);

  trayIcon.Icon := FInactiveIcon;
  trayIcon.PopUpMenu := pmTrayMenu;
  trayIcon.Visible := True;

  cmbLanguage.ItemIndex := FLanguageIdx;
  UpdateModelsCombo;
  UpdateUIStrings;

  timerStatus.Enabled := True;
  timerStatusTimer(nil);
end;

procedure TfrmManager.FormDestroy(Sender: TObject);
begin
  LogEvent('Manager encerrando.');
  if Assigned(FServerProcess) then
    FServerProcess.Free;
  FActiveIcon.Free;
  FInactiveIcon.Free;
end;

procedure TfrmManager.FormShow(Sender: TObject);
begin
  edtProjectPath.Text := FProjectPath;
  edtServerPath.Text := FChatServerPath;
  edtModelsPath.Text := FModelsPath;
  edtPort.Text := FPort;
  edtCtx.Text := FCtx;
  if cmbMaxNewTokens.Items.IndexOf(FMaxNewTokens) >= 0 then
    cmbMaxNewTokens.ItemIndex := cmbMaxNewTokens.Items.IndexOf(FMaxNewTokens)
  else
    cmbMaxNewTokens.ItemIndex := 2;
  chkAutoStart.Checked := FAutoStart;
  chkAutoOn.Checked := FAutoOn;
  chkUseGpu.Checked := FUseGpu;
  UpdateModelsCombo;
  cmbLanguage.ItemIndex := FLanguageIdx;
  UpdateUIStrings;
  pgcMain.ActivePage := tsConfig;
end;

procedure TfrmManager.btnBrowsePathClick(Sender: TObject);
begin
  selectDirDialog.InitialDir := edtProjectPath.Text;
  if selectDirDialog.Execute then
  begin
    edtProjectPath.Text := selectDirDialog.FileName;
    UpdateModelsCombo;
  end;
end;

procedure TfrmManager.btnBrowseServerClick(Sender: TObject);
var
  OpenDlg: TOpenDialog;
begin
  OpenDlg := TOpenDialog.Create(Self);
  try
    OpenDlg.Filter := 'ChatServer Binary (ChatServer.exe)|ChatServer.exe|All files (*.*)|*.*';
    OpenDlg.InitialDir := ExtractFilePath(edtServerPath.Text);
    if OpenDlg.Execute then
      edtServerPath.Text := OpenDlg.FileName;
  finally
    OpenDlg.Free;
  end;
end;

procedure TfrmManager.btnBrowseModelsClick(Sender: TObject);
begin
  selectDirDialog.InitialDir := edtModelsPath.Text;
  if selectDirDialog.Execute then
  begin
    edtModelsPath.Text := selectDirDialog.FileName;
    UpdateModelsCombo;
  end;
end;

procedure TfrmManager.btnSaveClick(Sender: TObject);
begin
  FProjectPath := edtProjectPath.Text;
  FChatServerPath := edtServerPath.Text;
  FModelsPath := edtModelsPath.Text;
  FPort := edtPort.Text;
  FCtx := edtCtx.Text;
  if Trim(FCtx) = '' then FCtx := '8192';
  if cmbMaxNewTokens.ItemIndex >= 0 then
    FMaxNewTokens := cmbMaxNewTokens.Items[cmbMaxNewTokens.ItemIndex]
  else
    FMaxNewTokens := '32';
  FAutoStart := chkAutoStart.Checked;
  FAutoOn := chkAutoOn.Checked;
  FUseGpu := chkUseGpu.Checked;
  if cmbModel.ItemIndex >= 0 then
    FSelectedModel := cmbModel.Items[cmbModel.ItemIndex];
  FLanguageIdx := cmbLanguage.ItemIndex;

  SaveConfig;
  UpdateUIStrings;
  Hide;
end;

procedure TfrmManager.btnCancelFormClick(Sender: TObject);
begin
  Hide;
end;

procedure TfrmManager.btnClearLogClick(Sender: TObject);
begin
  memLog.Lines.Clear;
  LogEvent('Log limpo pelo usuário.');
end;

procedure TfrmManager.cmbLanguageChange(Sender: TObject);
begin
  FLanguageIdx := cmbLanguage.ItemIndex;
  LogEvent('Idioma alterado no formulário para: ' + cmbLanguage.Items[FLanguageIdx]);
  UpdateUIStrings;
end;

procedure TfrmManager.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  CloseAction := caNone;
  Hide;
end;

procedure TfrmManager.menuStartClick(Sender: TObject);
begin
  StartServer;
end;

procedure TfrmManager.menuStopClick(Sender: TObject);
begin
  StopServer;
end;

procedure TfrmManager.menuConfigClick(Sender: TObject);
begin
  Show;
  WindowState := wsNormal;
  pgcMain.ActivePage := tsConfig;
  BringToFront;
end;

procedure TfrmManager.menuLogClick(Sender: TObject);
begin
  Show;
  WindowState := wsNormal;
  pgcMain.ActivePage := tsLog;
  BringToFront;
end;

procedure TfrmManager.menuExitClick(Sender: TObject);
begin
  LogEvent('Exit solicitado pelo menu da bandeja.');
  timerStatus.Enabled := False;
  StopServer;
  Application.Terminate;
end;

procedure TfrmManager.timerStatusTimer(Sender: TObject);
var
  Online: Boolean;
  SecsSinceLast: Double;
begin
  timerStatus.Enabled := False;
  try
    Online := IsServerOnline;
    if Online <> FServerActive then
    begin
      FServerActive := Online;
      if FServerActive then
        LogEvent('Status detectado: ONLINE')
      else
        LogEvent('Status detectado: OFFLINE');
      UpdateStatusUI;
    end;

    if FAutoOn and (not FServerActive) then
    begin
      SecsSinceLast := (Now - FLastStartAttempt) * 86400;
      if (FLastStartAttempt = 0) or (SecsSinceLast >= 60) then
      begin
        LogEvent('Auto On: ChatServer está offline. Tentando reiniciar (intervalo de 1 minuto)...');
        FLastStartAttempt := Now;
        StartServer;
      end;
    end;
  finally
    timerStatus.Enabled := True;
  end;
end;

procedure TfrmManager.trayIconDblClick(Sender: TObject);
begin
  Show;
  WindowState := wsNormal;
  pgcMain.ActivePage := tsConfig;
  BringToFront;
end;

procedure TfrmManager.btnBenchmarkClick(Sender: TObject);
begin
  if not Assigned(frmBenchmark) then
    frmBenchmark := TfrmBenchmark.Create(Application);
  frmBenchmark.SetLanguage(FLanguageIdx);
  frmBenchmark.Show;
  frmBenchmark.BringToFront;
end;

procedure TfrmManager.menuBenchmarkClick(Sender: TObject);
begin
  btnBenchmarkClick(Sender);
end;

procedure TfrmManager.btnChatClick(Sender: TObject);
var
  ActivePort, ActiveModel: string;
begin
  ActivePort := edtPort.Text;
  if ActivePort = '' then ActivePort := FPort;
  if ActivePort = '' then ActivePort := '8095';

  if cmbModel.ItemIndex >= 0 then
    ActiveModel := cmbModel.Items[cmbModel.ItemIndex]
  else
    ActiveModel := FSelectedModel;

  if not Assigned(frmChat) then
    frmChat := TfrmChat.Create(Application);

  frmChat.SetConfig(ActivePort, ActiveModel, FLanguageIdx);
  frmChat.Show;
  frmChat.BringToFront;
end;

procedure TfrmManager.menuChatClick(Sender: TObject);
begin
  btnChatClick(Sender);
end;

procedure TfrmManager.btnSetupClick(Sender: TObject);
var
  SetupPath: string;
  Proc: TProcess;
begin
  SetupPath := IncludeTrailingPathDelimiter(FProjectPath) + 'win_install' + PathDelim + 'installer.exe';

  if not FileExists(SetupPath) then
    SetupPath := IncludeTrailingPathDelimiter(FProjectPath) + 'win_install' + PathDelim + 'setup.exe';

  if not FileExists(SetupPath) then
    SetupPath := IncludeTrailingPathDelimiter(FProjectPath) + 'installer.exe';

  if not FileExists(SetupPath) then
    SetupPath := ExtractFilePath(Application.ExeName) + 'installer.exe';

  if not FileExists(SetupPath) then
    SetupPath := IncludeTrailingPathDelimiter(FProjectPath) + 'win_install' + PathDelim + 'installer.exe';

  if not FileExists(SetupPath) then
  begin
    LogEvent('Erro: Executável do Setup não encontrado em: ' + SetupPath);
    ShowMessage('Setup/Installer.exe não encontrado em ' + SetupPath + '.');
    Exit;
  end;

  LogEvent('Iniciando o Setup: ' + SetupPath);
  Proc := TProcess.Create(nil);
  try
    Proc.Executable := SetupPath;
    Proc.Execute;
  except
    on E: Exception do
    begin
      LogEvent('Erro ao iniciar o Setup: ' + E.Message);
      ShowMessage('Erro ao iniciar o Setup: ' + E.Message);
    end;
  end;
  Proc.Free;
end;

end.
