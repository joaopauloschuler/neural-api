unit mainform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, ComCtrls,
  StdCtrls, Buttons, FileUtil, CheckLst, Process, fphttpclient, opensslsockets, chatgpt, aibase, StrUtils;

type
  TDownloadItem = record
    FileName: string;
    URL: string;
  end;

  TModelInfo = record
    Name: string;
    Folder: string;
    Files: array of TDownloadItem;
  end;

  TfrmInstaller = class;

  { TModelDownloaderThread }
  TModelDownloaderThread = class(TThread)
  private
    FInstallerForm: TfrmInstaller;
    FDestDir: string;
    FModelInfo: TModelInfo;
    FIncremental: Boolean;
    FCurrentFile: string;
    FStatusMessage: string;
    FLogLine: string;
    FErrorOccurred: Boolean;
    FProcess: TProcess;
    FReport: string;
    procedure ShowStatus;
    procedure SyncLog;
    procedure DoFinished;
  protected
    procedure Execute; override;
  public
    constructor Create(AForm: TfrmInstaller; const ADestDir: string; const AModel: TModelInfo; AIncremental: Boolean);
    destructor Destroy; override;
    procedure CancelDownload;
  end;

  { TChatTestThread }
  TChatTestThread = class(TThread)
  private
    FInstallerForm: TfrmInstaller;
    FProvider: TAIProvider;
    FToken: string;
    FURL: string;
    FCustomModel: string;
    FQuestion: string;
    FResponseText: string;
    FSuccess: Boolean;
    procedure UpdateUI;
  protected
    procedure Execute; override;
  public
    constructor Create(AForm: TfrmInstaller; AProvider: TAIProvider; const AToken, AURL, ACustomModel, AQuestion: string);
  end;

  { TChatServerThread }
  TChatServerThread = class(TThread)
  private
    FInstallerForm: TfrmInstaller;
    FDestDir: string;
    FModelFolder: string;
    FPort: string;
    FLogLine: string;
    FStatusMessage: string;
    FSuccess: Boolean;
    FProcess: TProcess;
    FHttpClient: TFPHttpClient;
    procedure SyncLog;
    procedure SyncStatus;
    procedure DoFinished;
    function RunProcess(const ABinary: string; AParams: array of string; ARedirect: Boolean): Boolean;
    function CheckEndpoint: Boolean;
    procedure ReadServerOutput;
  protected
    procedure Execute; override;
  public
    constructor Create(AForm: TfrmInstaller; const ADestDir, AModelFolder, APort: string);
    destructor Destroy; override;
    procedure StopServer;
  end;

  { TfrmDownloadReport }
  TfrmDownloadReport = class(TForm)
  private
    FMemo: TMemo;
    FBtnClose: TButton;
  public
    constructor Create(AOwner: TComponent); override;
    property Memo: TMemo read FMemo;
  end;

  { TfrmInstaller }

  TfrmInstaller = class(TForm)
    btnCancel: TButton;
    btnBack: TButton;
    btnNext: TButton;
    btnBrowse: TButton;
    btnDownload: TButton;
    btnSendQuestion: TButton;
    cmbProvider: TComboBox;
    cmbModels: TComboBox;
    edtDestDir: TEdit;
    edtToken: TEdit;
    edtURL: TEdit;
    edtCustomModel: TEdit;
    lblTitle: TLabel;
    lblWelcome: TLabel;
    lblWelcomeDesc: TLabel;
    lblDestDir: TLabel;
    lblDownloadStatus: TLabel;
    lblProvider: TLabel;
    lblToken: TLabel;
    lblURL: TLabel;
    lblCustomModel: TLabel;
    lblQuestion: TLabel;
    lblResponse: TLabel;
    memWelcomeInfo: TMemo;
    memQuestion: TMemo;
    memResponse: TMemo;
    pgcWizard: TPageControl;
    pnlSidebar: TPanel;
    pnlButtons: TPanel;
    pnlMain: TPanel;
    pbDownload: TProgressBar;
    selectDirDialog: TSelectDirectoryDialog;
    tsWelcome: TTabSheet;
    tsDirectory: TTabSheet;
    tsDownloader: TTabSheet;
    tsChatGPT: TTabSheet;
    tsFinished: TTabSheet;
    chkModelFiles: TCheckListBox;
    lblSelectFiles: TLabel;
    lblStepWelcome: TLabel;
    lblStepPath: TLabel;
    lblStepModels: TLabel;
    lblStepTest: TLabel;
    lblStepFinish: TLabel;
    lblFinishTitle: TLabel;
    lblFinishDesc: TLabel;
    lblLanguage: TLabel;
    cmbLanguage: TComboBox;
    memDownloadLog: TMemo;
    tsChatServer: TTabSheet;
    lblServerStatus: TLabel;
    btnStartServer: TButton;
    memServerLog: TMemo;
    lblStepServer: TLabel;
    lblServerPort: TLabel;
    edtServerPort: TEdit;
    lblLazarusDir: TLabel;
    edtLazarusDir: TEdit;
    btnBrowseLazarus: TButton;
    procedure btnBackClick(Sender: TObject);
    procedure btnBrowseClick(Sender: TObject);
    procedure btnBrowseLazarusClick(Sender: TObject);
    procedure btnCancelClick(Sender: TObject);
    procedure btnDownloadClick(Sender: TObject);
    procedure btnNextClick(Sender: TObject);
    procedure btnSendQuestionClick(Sender: TObject);
    procedure cmbModelsChange(Sender: TObject);
    procedure cmbProviderChange(Sender: TObject);
    procedure cmbLanguageChange(Sender: TObject);
    procedure btnStartServerClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    FCurrentPage: Integer;
    FDownloaderThread: TModelDownloaderThread;
    FChatThread: TChatTestThread;
    FChatServerThread: TObject; // Declarado como TObject temporariamente para evitar dependência de ordem
    FModelsList: array of TModelInfo;
    FDownloadReport: string;
    FDownloadAllowedNext: Boolean;
    FServerAllowedNext: Boolean;
    procedure UpdateLanguageUI(ALang: Integer);
    procedure UpdateWizardState;
    procedure SetupModelsList;
    procedure HighlightSidebarStep(AStepIndex: Integer);
  public
    procedure OnDownloadProgress(const AFileName: string; AMax, AVal: Int64);
    procedure OnDownloadStatus(const AMsg: string; AIsError: Boolean);
    procedure OnDownloadFinished(ASuccess: Boolean; const AReport: string);
    procedure OnDownloadLog(const AMsg: string);
    procedure OnServerActive(ASuccess: Boolean; const AMsg: string);
    procedure OnServerLog(const AMsg: string);
    procedure OnChatFinished(ASuccess: Boolean; const AResponse: string);
  end;

var
  frmInstaller: TfrmInstaller;

implementation

{$R *.lfm}

{ FormatSize helper }
function FormatSize(Bytes: Int64): string;
begin
  if Bytes > 1024 * 1024 then
    Result := Format('%.2f MB', [Bytes / (1024.0 * 1024.0)])
  else if Bytes > 1024 then
    Result := Format('%.2f KB', [Bytes / 1024.0])
  else
    Result := IntToStr(Bytes) + ' B';
end;

{ TfrmDownloadReport }

constructor TfrmDownloadReport.Create(AOwner: TComponent);
begin
  inherited CreateNew(AOwner);
  Caption := 'Relatório de Instalação / Installation Report';
  Width := 500;
  Height := 350;
  Position := poScreenCenter;
  
  FMemo := TMemo.Create(Self);
  FMemo.Parent := Self;
  FMemo.Align := alClient;
  FMemo.ReadOnly := True;
  FMemo.ScrollBars := ssAutoVertical;
  
  FBtnClose := TButton.Create(Self);
  FBtnClose.Parent := Self;
  FBtnClose.Align := alBottom;
  FBtnClose.Height := 40;
  FBtnClose.Caption := 'Fechar / Close';
  FBtnClose.ModalResult := mrOk;
end;

{ TModelDownloaderThread }

constructor TModelDownloaderThread.Create(AForm: TfrmInstaller; const ADestDir: string; const AModel: TModelInfo; AIncremental: Boolean);
begin
  inherited Create(True);
  FInstallerForm := AForm;
  FDestDir := ADestDir;
  FModelInfo := AModel;
  FIncremental := AIncremental;
  FreeOnTerminate := True;
end;

destructor TModelDownloaderThread.Destroy;
begin
  if Assigned(FProcess) then
  begin
    if FProcess.Active then
      FProcess.Terminate(0);
    FProcess.Free;
  end;
  inherited Destroy;
end;

procedure TModelDownloaderThread.CancelDownload;
begin
  Terminate;
  if Assigned(FProcess) and FProcess.Active then
    FProcess.Terminate(0);
end;

procedure TModelDownloaderThread.ShowStatus;
begin
  if not Terminated then
    FInstallerForm.OnDownloadStatus(FStatusMessage, FErrorOccurred);
end;

procedure TModelDownloaderThread.SyncLog;
begin
  if not Terminated then
    FInstallerForm.OnDownloadLog(FLogLine);
end;

procedure TModelDownloaderThread.Execute;
var
  I: Integer;
  DestFolder: string;
  DestFilePath: string;
  Item: TDownloadItem;
begin
  FErrorOccurred := False;
  DestFolder := IncludeTrailingPathDelimiter(FDestDir) + 'models' + PathDelim + FModelInfo.Folder;
  FReport := '=== RELATÓRIO DE INSTALAÇÃO DE MODELOS ===' + #13#10 +
             'Modelo: ' + FModelInfo.Name + #13#10 +
             'Pasta de Destino: ' + DestFolder + #13#10 +
             'Modo de Download: ' + IfThen(FIncremental, 'Incremental', 'Sobrescrever/Novo') + #13#10#13#10;
             
  FLogLine := 'Iniciando processo de download...';
  Synchronize(@SyncLog);
  
  try
    FLogLine := 'Criando diretório: ' + DestFolder;
    Synchronize(@SyncLog);
    
    if not ForceDirectories(DestFolder) then
    begin
      FStatusMessage := 'Erro: Não foi possível criar pasta ' + DestFolder;
      FLogLine := FStatusMessage;
      FErrorOccurred := True;
      Synchronize(@ShowStatus);
      Synchronize(@SyncLog);
      Exit;
    end;

    for I := 0 to Length(FModelInfo.Files) - 1 do
    begin
      if Terminated then Break;

      Item := FModelInfo.Files[I];
      FCurrentFile := Item.FileName;
      DestFilePath := IncludeTrailingPathDelimiter(DestFolder) + Item.FileName;

      FStatusMessage := 'Baixando: ' + Item.FileName;
      Synchronize(@ShowStatus);

      // Verificação incremental
      if FIncremental and FileExists(DestFilePath) then
      begin
        FLogLine := '[INCREMENTAL] Arquivo já existe, pulando: ' + Item.FileName;
        FReport := FReport + '- ' + Item.FileName + ': Ignorado (Incremental) (' + FormatSize(FileUtil.FileSize(DestFilePath)) + ')' + #13#10;
        Synchronize(@SyncLog);
        Continue;
      end;

      if not FIncremental and FileExists(DestFilePath) then
      begin
        FLogLine := '[SOBRESCREVER] Apagando arquivo existente: ' + Item.FileName;
        Synchronize(@SyncLog);
        DeleteFile(DestFilePath);
      end;

      FLogLine := '[DOWNLOAD] Baixando arquivo: ' + Item.FileName + ' via curl...';
      Synchronize(@SyncLog);

      FProcess := TProcess.Create(nil);
      try
        FProcess.Executable := 'curl.exe';
        FProcess.Parameters.Add('-L');
        FProcess.Parameters.Add('-o');
        FProcess.Parameters.Add(DestFilePath);
        FProcess.Parameters.Add(Item.URL);
        FProcess.Options := [poNewConsole];

        try
          FProcess.Execute;
          
          while FProcess.Active and not Terminated do
          begin
            Sleep(100);
          end;
          
          if Terminated then
          begin
            if FProcess.Active then
              FProcess.Terminate(0);
            Break;
          end;

          if FProcess.ExitStatus <> 0 then
            raise Exception.Create('curl retornou código de erro ' + IntToStr(FProcess.ExitStatus));
            
          FLogLine := '[DOWNLOAD] Concluído: ' + Item.FileName;
          FReport := FReport + '- ' + Item.FileName + ': Baixado (' + FormatSize(FileUtil.FileSize(DestFilePath)) + ')' + #13#10;
          Synchronize(@SyncLog);
        except
          on E: Exception do
          begin
            FStatusMessage := 'Erro baixando ' + Item.FileName + ': ' + E.Message;
            FLogLine := '[ERRO] ' + FStatusMessage;
            FReport := FReport + '- ' + Item.FileName + ': Falhou (' + E.Message + ')' + #13#10;
            FErrorOccurred := True;
            Synchronize(@ShowStatus);
            Synchronize(@SyncLog);
            Break;
          end;
        end;
      finally
        FProcess.Free;
        FProcess := nil;
      end;
    end;

  finally
    // Nenhuma limpeza de HttpClient necessária agora
  end;

  if not Terminated and not FErrorOccurred then
  begin
    FStatusMessage := 'Downloads concluídos com sucesso!';
    FLogLine := FStatusMessage;
    Synchronize(@ShowStatus);
    Synchronize(@SyncLog);
  end;

  Synchronize(@DoFinished);
end;

procedure TModelDownloaderThread.DoFinished;
begin
  FInstallerForm.OnDownloadFinished(not FErrorOccurred and not Terminated, FReport);
end;

{ TChatTestThread }

constructor TChatTestThread.Create(AForm: TfrmInstaller; AProvider: TAIProvider; const AToken, AURL, ACustomModel, AQuestion: string);
begin
  inherited Create(True);
  FInstallerForm := AForm;
  FProvider := AProvider;
  FToken := AToken;
  FURL := AURL;
  FCustomModel := ACustomModel;
  FQuestion := AQuestion;
  FreeOnTerminate := True;
end;

procedure TChatTestThread.UpdateUI;
begin
  FInstallerForm.OnChatFinished(FSuccess, FResponseText);
end;

procedure TChatTestThread.Execute;
var
  Chat: TCHATGPT;
begin
  FSuccess := False;
  FResponseText := '';
  
  Chat := TCHATGPT.Create(nil);
  try
    Chat.Provider := FProvider;
    Chat.TOKEN := FToken;
    if FURL <> '' then
      Chat.URL := FURL;
    if FCustomModel <> '' then
      Chat.CustomModel := FCustomModel;
    
    // Escolhe um modelo adequado com base no Provider
    case FProvider of
      AIP_OPENAI: Chat.TipoChat := VCT_GPT4O_MINI;
      AIP_GEMINI: Chat.TipoChat := VCT_GEMINI_15_FLASH;
      AIP_LOCAL: Chat.TipoChat := VCT_CUSTOM;
      AIP_OPENROUTER: Chat.TipoChat := VCT_OPENROUTER_DEEPSEEK_R1_FREE;
      else Chat.TipoChat := VCT_CUSTOM;
    end;

    try
      FSuccess := Chat.SendQuestion(FQuestion);
      if FSuccess then
        FResponseText := Chat.Response
      else
        FResponseText := Chat.LastError;
    except
      on E: Exception do
      begin
        FSuccess := False;
        FResponseText := 'Erro na execução: ' + E.Message;
      end;
    end;
  finally
    Chat.Free;
  end;

  Synchronize(@UpdateUI);
end;

{ TChatServerThread }

constructor TChatServerThread.Create(AForm: TfrmInstaller; const ADestDir, AModelFolder, APort: string);
begin
  inherited Create(True);
  FInstallerForm := AForm;
  FDestDir := ADestDir;
  FModelFolder := AModelFolder;
  FPort := APort;
  FreeOnTerminate := True;
end;

destructor TChatServerThread.Destroy;
begin
  if Assigned(FHttpClient) then
    FHttpClient.Free;
  if Assigned(FProcess) then
  begin
    if FProcess.Active then
      FProcess.Terminate(0);
    FProcess.Free;
  end;
  inherited Destroy;
end;

procedure TChatServerThread.StopServer;
begin
  Terminate;
  if Assigned(FProcess) and FProcess.Active then
    FProcess.Terminate(0);
end;

procedure TChatServerThread.SyncLog;
begin
  if not Terminated then
    FInstallerForm.OnServerLog(FLogLine);
end;

procedure TChatServerThread.SyncStatus;
begin
  if not Terminated then
    FInstallerForm.OnServerActive(FSuccess, FStatusMessage);
end;

procedure TChatServerThread.DoFinished;
begin
  if not Terminated then
    FInstallerForm.OnServerActive(FSuccess, FStatusMessage);
end;

function TChatServerThread.RunProcess(const ABinary: string; AParams: array of string; ARedirect: Boolean): Boolean;
var
  Buffer: array[0..1023] of Byte;
  BytesRead: LongInt;
  Line: string;
  I: Integer;
begin
  Result := False;
  FProcess := TProcess.Create(nil);
  try
    FProcess.Executable := ABinary;
    for I := 0 to High(AParams) do
      FProcess.Parameters.Add(AParams[I]);
    FProcess.CurrentDirectory := ExtractFilePath(ABinary);
    
    if ARedirect then
      FProcess.Options := [poUsePipes, poNoConsole]
    else
      FProcess.Options := [];

    try
      FProcess.Execute;
      Result := True;
    except
      on E: Exception do
      begin
        FLogLine := '[ERRO] Falha ao executar ' + ABinary + ': ' + E.Message;
        Synchronize(@SyncLog);
        Exit(False);
      end;
    end;

    if ARedirect then
    begin
      while FProcess.Active and not Terminated do
      begin
        if FProcess.Output.NumBytesAvailable > 0 then
        begin
          BytesRead := FProcess.Output.Read(Buffer, SizeOf(Buffer) - 1);
          if BytesRead > 0 then
          begin
            Buffer[BytesRead] := 0;
            Line := PChar(@Buffer[0]);
            FLogLine := Line;
            Synchronize(@SyncLog);
          end;
        end;
        Sleep(50);
      end;
    end;
  finally
    if not ARedirect then
    begin
      // Se não redireciona, mantemos o processo rodando em background e gerenciado pelo Destroy
    end
    else
    begin
      FProcess.Free;
      FProcess := nil;
    end;
  end;
end;

function TChatServerThread.CheckEndpoint: Boolean;
var
  Res: string;
begin
  Result := False;
  if not Assigned(FHttpClient) then
  begin
    FHttpClient := TFPHttpClient.Create(nil);
    FHttpClient.ConnectTimeout := 2;
    FHttpClient.IOTimeout := 2;
  end;
  try
    Res := FHttpClient.Get('http://127.0.0.1:' + FPort + '/v1/models');
    if (Pos('qwen', LowerCase(Res)) > 0) or (Pos('llama', LowerCase(Res)) > 0) or (Pos('smol', LowerCase(Res)) > 0) or (Pos('model', LowerCase(Res)) > 0) then
      Result := True
    else
      Result := (Res <> ''); // Se respondeu qualquer coisa, está rodando
  except
    // Falhou em conectar
  end;
end;

procedure TChatServerThread.ReadServerOutput;
var
  Buffer: array[0..4095] of Byte;
  BytesRead: LongInt;
  Line: string;
begin
  if not Assigned(FProcess) then Exit;
  
  // Read stdout
  if Assigned(FProcess.Output) then
  begin
    while FProcess.Output.NumBytesAvailable > 0 do
    begin
      BytesRead := FProcess.Output.Read(Buffer, SizeOf(Buffer) - 1);
      if BytesRead > 0 then
      begin
        Buffer[BytesRead] := 0;
        Line := PChar(@Buffer[0]);
        FLogLine := Line;
        Synchronize(@SyncLog);
      end;
    end;
  end;

  // Read stderr
  if Assigned(FProcess.Stderr) then
  begin
    while FProcess.Stderr.NumBytesAvailable > 0 do
    begin
      BytesRead := FProcess.Stderr.Read(Buffer, SizeOf(Buffer) - 1);
      if BytesRead > 0 then
      begin
        Buffer[BytesRead] := 0;
        Line := PChar(@Buffer[0]);
        FLogLine := Line;
        Synchronize(@SyncLog);
      end;
    end;
  end;
end;

procedure TChatServerThread.Execute;
var
  LazBuildPath: string;
  ServerLpiPath: string;
  ServerExePath: string;
  ModelPath: string;
  I, J: Integer;
begin
  FSuccess := False;
  FStatusMessage := '';

  // 1. Compilar o ChatServer
  LazBuildPath := IncludeTrailingPathDelimiter(FInstallerForm.edtLazarusDir.Text) + 'lazbuild.exe';
  ServerLpiPath := IncludeTrailingPathDelimiter(FDestDir) + 'examples' + PathDelim + 'ChatTerminal' + PathDelim + 'ChatServer.lpi';
  ServerExePath := IncludeTrailingPathDelimiter(FDestDir) + 'examples' + PathDelim + 'ChatTerminal' + PathDelim + 'ChatServer.exe';

  FLogLine := '[COMPILAÇÃO] Iniciando compilação do ChatServer...';
  Synchronize(@SyncLog);

  if not FileExists(LazBuildPath) then
  begin
    FLogLine := '[ERRO] compilador lazbuild.exe não encontrado em ' + LazBuildPath;
    FStatusMessage := 'Compilador não encontrado.';
    Synchronize(@SyncLog);
    Synchronize(@SyncStatus);
    Exit;
  end;

  if not FileExists(ServerLpiPath) then
  begin
    FLogLine := '[ERRO] Projeto do ChatServer não encontrado em ' + ServerLpiPath;
    FStatusMessage := 'Projeto não encontrado.';
    Synchronize(@SyncLog);
    Synchronize(@SyncStatus);
    Exit;
  end;

  FStatusMessage := 'Compilando ChatServer...';
  Synchronize(@SyncStatus);

  if not RunProcess(LazBuildPath, ['--cpu=x86_64', '--os=win64', ServerLpiPath], True) then
  begin
    FStatusMessage := 'Falha na compilação.';
    Synchronize(@SyncStatus);
    Exit;
  end;

  // Resolve executable path
  if not FileExists(ServerExePath) then
    ServerExePath := IncludeTrailingPathDelimiter(FDestDir) + 'bin' + PathDelim + 'x86_64-win64' + PathDelim + 'bin' + PathDelim + 'ChatServer.exe';

  if not FileExists(ServerExePath) then
  begin
    FLogLine := '[ERRO] Executável do ChatServer não encontrado em ' + ServerExePath;
    FStatusMessage := 'Executável não encontrado.';
    Synchronize(@SyncLog);
    Synchronize(@SyncStatus);
    Exit;
  end;

  FLogLine := '[COMPILAÇÃO] ChatServer compilado com sucesso!';
  Synchronize(@SyncLog);

  // 2. Iniciar o ChatServer em background
  ModelPath := IncludeTrailingPathDelimiter(FDestDir) + 'models' + PathDelim + FModelFolder;
  FLogLine := '[EXECUÇÃO] Iniciando ChatServer com o modelo: ' + ModelPath;
  Synchronize(@SyncLog);

  FStatusMessage := 'Iniciando servidor...';
  Synchronize(@SyncStatus);

  // Executa o servidor em background (sem redirecionar output para não travar a thread e manter o servidor ativo)
  if not RunProcess(ServerExePath, [ModelPath, '--port', FPort], False) then
  begin
    FStatusMessage := 'Falha ao iniciar ChatServer.exe.';
    Synchronize(@SyncStatus);
    Exit;
  end;

  // 3. Monitorar o endpoint http://localhost:<FPort>/v1/models
  FLogLine := '[MONITOR] Aguardando o servidor responder na porta ' + FPort + '...';
  Synchronize(@SyncLog);

  for I := 1 to 20 do // Tentar por 20 segundos
  begin
    if Terminated then Break;

    ReadServerOutput;

    if not FProcess.Active then
    begin
      ReadServerOutput;
      FLogLine := '[ERRO] O processo ChatServer.exe finalizou com código de saída: ' + IntToStr(FProcess.ExitStatus);
      Synchronize(@SyncLog);
      FStatusMessage := 'Servidor finalizou inesperadamente.';
      Synchronize(@SyncStatus);
      Break;
    end;
    
    if CheckEndpoint then
    begin
      FLogLine := '[MONITOR] Servidor respondeu com sucesso! PID: ' + IntToStr(FProcess.ProcessID);
      FLogLine := FLogLine + '. Aguardando 5 segundos para verificar estabilidade...';
      Synchronize(@SyncLog);
      
      // Aguarda 5 segundos lendo a saída continuamente para manter o log atualizado e evitar travamento
      for J := 1 to 50 do
      begin
        if Terminated then Break;
        Sleep(100);
        ReadServerOutput;
      end;

      if not FProcess.Active then
      begin
        ReadServerOutput;
        FLogLine := '[ERRO] O processo ChatServer.exe caiu logo após subir. Código de saída: ' + IntToStr(FProcess.ExitStatus);
        Synchronize(@SyncLog);
        FStatusMessage := 'Servidor caiu após iniciar.';
        Synchronize(@SyncStatus);
        Break;
      end;

      if CheckEndpoint then
      begin
        FSuccess := True;
        FStatusMessage := 'Serviço Ativo (Porta ' + FPort + ')';
        FLogLine := '[MONITOR] Servidor está de pé e estável!';
        Synchronize(@SyncLog);
        Synchronize(@SyncStatus);
        Break;
      end
      else
      begin
        FLogLine := '[ERRO] Servidor parou de responder após 5 segundos.';
        Synchronize(@SyncLog);
      end;
    end;
    
    FLogLine := '[MONITOR] Tentativa ' + IntToStr(I) + '/20...';
    Synchronize(@SyncLog);
    
    // Sleep 1 second reading output
    for J := 1 to 10 do
    begin
      if Terminated then Break;
      Sleep(100);
      ReadServerOutput;
    end;
  end;

  if FSuccess then
  begin
    while not Terminated do
    begin
      ReadServerOutput;
      if not FProcess.Active then
      begin
        ReadServerOutput;
        FSuccess := False;
        FStatusMessage := 'Servidor caiu inesperadamente.';
        FLogLine := '[ERRO] O processo ChatServer.exe parou de rodar. Código de saída: ' + IntToStr(FProcess.ExitStatus);
        Synchronize(@SyncLog);
        Synchronize(@SyncStatus);
        Break;
      end;
      Sleep(500);
    end;
  end;

  if not FSuccess and not Terminated then
  begin
    if FStatusMessage = '' then
      FStatusMessage := 'Timeout ao iniciar o servidor.';
    if FLogLine = '' then
      FLogLine := '[ERRO] Servidor não respondeu a tempo na porta ' + FPort + '.';
    Synchronize(@SyncLog);
    Synchronize(@SyncStatus);
  end;

  Synchronize(@DoFinished);
end;

{ TfrmInstaller }

procedure TfrmInstaller.FormCreate(Sender: TObject);
begin
  FCurrentPage := 0;
  edtDestDir.Text := ExtractFilePath(ExcludeTrailingPathDelimiter(ExtractFileDir(Application.ExeName)));
  if edtDestDir.Text = '' then
    edtDestDir.Text := 'D:\projetos\neural-api';
    
  edtLazarusDir.Text := 'C:\lazarus';
    
  SetupModelsList;
  UpdateLanguageUI(cmbLanguage.ItemIndex);
  UpdateWizardState;
end;

procedure TfrmInstaller.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if Assigned(FDownloaderThread) then
    FDownloaderThread.CancelDownload;
  if Assigned(FChatServerThread) then
    TChatServerThread(FChatServerThread).StopServer;
end;

procedure TfrmInstaller.SetupModelsList;
var
  Qwen, Llama, Smol: TModelInfo;
begin
  SetLength(FModelsList, 3);

  // Qwen 2.5 0.5B Instruct
  Qwen.Name := 'Qwen 2.5 0.5B Instruct (Recomendado)';
  Qwen.Folder := 'Qwen2.5-0.5B-Instruct';
  SetLength(Qwen.Files, 5);
  Qwen.Files[0].FileName := 'config.json';
  Qwen.Files[0].URL := 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json';
  Qwen.Files[1].FileName := 'tokenizer.json';
  Qwen.Files[1].URL := 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json';
  Qwen.Files[2].FileName := 'tokenizer_config.json';
  Qwen.Files[2].URL := 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json';
  Qwen.Files[3].FileName := 'generation_config.json';
  Qwen.Files[3].URL := 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/generation_config.json';
  Qwen.Files[4].FileName := 'model.safetensors';
  Qwen.Files[4].URL := 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors';
  FModelsList[0] := Qwen;

  // TinyLlama 1.1B Chat
  Llama.Name := 'TinyLlama 1.1B Chat';
  Llama.Folder := 'TinyLlama-1.1B-Chat-v1.0';
  SetLength(Llama.Files, 4);
  Llama.Files[0].FileName := 'config.json';
  Llama.Files[0].URL := 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json';
  Llama.Files[1].FileName := 'tokenizer.json';
  Llama.Files[1].URL := 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json';
  Llama.Files[2].FileName := 'tokenizer_config.json';
  Llama.Files[2].URL := 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json';
  Llama.Files[3].FileName := 'model.safetensors';
  Llama.Files[3].URL := 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors';
  FModelsList[1] := Llama;

  // SmolLM2 1.7B Instruct
  Smol.Name := 'SmolLM2 1.7B Instruct';
  Smol.Folder := 'SmolLM2-1.7B-Instruct';
  SetLength(Smol.Files, 4);
  Smol.Files[0].FileName := 'config.json';
  Smol.Files[0].URL := 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/config.json';
  Smol.Files[1].FileName := 'tokenizer.json';
  Smol.Files[1].URL := 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/tokenizer.json';
  Smol.Files[2].FileName := 'tokenizer_config.json';
  Smol.Files[2].URL := 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/tokenizer_config.json';
  Smol.Files[3].FileName := 'model.safetensors';
  Smol.Files[3].URL := 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/model.safetensors';
  FModelsList[2] := Smol;

  cmbModels.Items.Clear;
  cmbModels.Items.Add(Qwen.Name);
  cmbModels.Items.Add(Llama.Name);
  cmbModels.Items.Add(Smol.Name);
  cmbModels.ItemIndex := 0;
  cmbModelsChange(nil);
end;

procedure TfrmInstaller.cmbModelsChange(Sender: TObject);
var
  Idx, I: Integer;
begin
  Idx := cmbModels.ItemIndex;
  if (Idx >= 0) and (Idx < Length(FModelsList)) then
  begin
    chkModelFiles.Items.Clear;
    for I := 0 to Length(FModelsList[Idx].Files) - 1 do
    begin
      chkModelFiles.Items.Add(FModelsList[Idx].Files[I].FileName);
      chkModelFiles.Checked[I] := True; // Padrão marcar todos
    end;
    
    // Atualiza o simulador de IA para usar Ollama local e o modelo selecionado
    cmbProvider.ItemIndex := 3; // Local Ollama
    cmbProviderChange(nil);
    edtCustomModel.Text := FModelsList[Idx].Folder;
  end;
end;

procedure TfrmInstaller.cmbProviderChange(Sender: TObject);
begin
  // Atualiza URLs padrão e descrições com base no provider
  case cmbProvider.ItemIndex of
    0: { OpenAI }
      begin
        edtURL.Text := 'https://api.openai.com/v1/chat/completions';
        edtCustomModel.Text := 'gpt-4o-mini';
      end;
    1: { OpenRouter }
      begin
        edtURL.Text := 'https://openrouter.ai/api/v1/chat/completions';
        edtCustomModel.Text := 'google/gemini-2.5-flash:free';
      end;
    3: { Local / Ollama }
      begin
        edtURL.Text := 'http://localhost:11434';
        edtCustomModel.Text := 'deepseek-r1:8b';
      end;
    4: { Gemini }
      begin
        edtURL.Text := 'https://generativelanguage.googleapis.com/v1beta/models';
        edtCustomModel.Text := 'gemini-1.5-flash';
      end;
  end;
end;

procedure TfrmInstaller.HighlightSidebarStep(AStepIndex: Integer);
begin
  // Reset colors
  lblStepWelcome.Font.Style := [];
  lblStepPath.Font.Style := [];
  lblStepModels.Font.Style := [];
  lblStepServer.Font.Style := [];
  lblStepTest.Font.Style := [];
  lblStepFinish.Font.Style := [];
  
  lblStepWelcome.Font.Color := clGray;
  lblStepPath.Font.Color := clGray;
  lblStepModels.Font.Color := clGray;
  lblStepServer.Font.Color := clGray;
  lblStepTest.Font.Color := clGray;
  lblStepFinish.Font.Color := clGray;

  case AStepIndex of
    0:
      begin
        lblStepWelcome.Font.Style := [fsBold];
        lblStepWelcome.Font.Color := clWhite;
      end;
    1:
      begin
        lblStepPath.Font.Style := [fsBold];
        lblStepPath.Font.Color := clWhite;
      end;
    2:
      begin
        lblStepModels.Font.Style := [fsBold];
        lblStepModels.Font.Color := clWhite;
      end;
    3:
      begin
        lblStepServer.Font.Style := [fsBold];
        lblStepServer.Font.Color := clWhite;
      end;
    4:
      begin
        lblStepTest.Font.Style := [fsBold];
        lblStepTest.Font.Color := clWhite;
      end;
    5:
      begin
        lblStepFinish.Font.Style := [fsBold];
        lblStepFinish.Font.Color := clWhite;
      end;
  end;
end;

procedure TfrmInstaller.UpdateWizardState;
begin
  pgcWizard.ActivePageIndex := FCurrentPage;
  HighlightSidebarStep(FCurrentPage);

  btnBack.Enabled := FCurrentPage > 0;
  
  // Controle de permissão de avanço para cada etapa
  case FCurrentPage of
    2: btnNext.Enabled := FDownloadAllowedNext; // Exige download finalizado e relatório fechado
    3: btnNext.Enabled := FServerAllowedNext;   // Exige servidor ativo
    4:
      begin
        btnNext.Enabled := True;
        if cmbProvider.ItemIndex = 3 then // Local Ollama
          edtURL.Text := 'http://localhost:' + edtServerPort.Text;
      end;
    else btnNext.Enabled := True;
  end;
  
  if FCurrentPage = pgcWizard.PageCount - 1 then
  begin
    btnNext.Caption := 'Concluir';
    btnCancel.Enabled := False;
  end
  else
  begin
    btnNext.Caption := 'Avançar >';
    btnCancel.Enabled := True;
  end;
end;

procedure TfrmInstaller.btnNextClick(Sender: TObject);
begin
  if FCurrentPage = pgcWizard.PageCount - 1 then
  begin
    Close;
  end
  else
  begin
    Inc(FCurrentPage);
    UpdateWizardState;
  end;
end;

procedure TfrmInstaller.btnBackClick(Sender: TObject);
begin
  if FCurrentPage > 0 then
  begin
    Dec(FCurrentPage);
    UpdateWizardState;
  end;
end;

procedure TfrmInstaller.btnBrowseClick(Sender: TObject);
begin
  selectDirDialog.InitialDir := edtDestDir.Text;
  if selectDirDialog.Execute then
  begin
    edtDestDir.Text := selectDirDialog.FileName;
  end;
end;

procedure TfrmInstaller.btnBrowseLazarusClick(Sender: TObject);
begin
  selectDirDialog.InitialDir := edtLazarusDir.Text;
  if selectDirDialog.Execute then
  begin
    edtLazarusDir.Text := selectDirDialog.FileName;
  end;
end;

procedure TfrmInstaller.btnCancelClick(Sender: TObject);
begin
  if MessageDlg('Deseja realmente cancelar a instalação?', mtConfirmation, [mbYes, mbNo], 0) = mrYes then
  begin
    Close;
  end;
end;

procedure TfrmInstaller.btnDownloadClick(Sender: TObject);
var
  Idx, I: Integer;
  SelectedModel: TModelInfo;
  ActiveFilesCount: Integer;
  DestFolder: string;
  HasExisting: Boolean;
  Incremental: Boolean;
begin
  if Assigned(FDownloaderThread) then
  begin
    FDownloaderThread.CancelDownload;
    btnDownload.Caption := 'Baixar Modelos Selecionados';
    Exit;
  end;

  Idx := cmbModels.ItemIndex;
  if (Idx < 0) or (Idx >= Length(FModelsList)) then Exit;

  // Filtrar arquivos marcados
  SelectedModel.Name := FModelsList[Idx].Name;
  SelectedModel.Folder := FModelsList[Idx].Folder;
  
  ActiveFilesCount := 0;
  for I := 0 to chkModelFiles.Items.Count - 1 do
    if chkModelFiles.Checked[I] then
      Inc(ActiveFilesCount);

  if ActiveFilesCount = 0 then
  begin
    ShowMessage('Selecione ao menos um arquivo para baixar!');
    Exit;
  end;

  SetLength(SelectedModel.Files, ActiveFilesCount);
  ActiveFilesCount := 0;
  for I := 0 to chkModelFiles.Items.Count - 1 do
  begin
    if chkModelFiles.Checked[I] then
    begin
      SelectedModel.Files[ActiveFilesCount] := FModelsList[Idx].Files[I];
      Inc(ActiveFilesCount);
    end;
  end;

  // Verificação prévia se algum arquivo já existe
  DestFolder := IncludeTrailingPathDelimiter(edtDestDir.Text) + 'models' + PathDelim + FModelsList[Idx].Folder;
  HasExisting := False;
  for I := 0 to chkModelFiles.Items.Count - 1 do
  begin
    if chkModelFiles.Checked[I] then
    begin
      if FileExists(IncludeTrailingPathDelimiter(DestFolder) + FModelsList[Idx].Files[I].FileName) then
      begin
        HasExisting := True;
        Break;
      end;
    end;
  end;

  Incremental := False;
  if HasExisting then
  begin
    case QuestionDlg('Aviso de Arquivos Existentes',
                    'Alguns arquivos do modelo já existem na pasta de destino.' + #13#10 +
                    'Deseja sobrescrever os arquivos existentes ou realizar download incremental?',
                    mtConfirmation,
                    [mrYes, 'Sobrescrever', mrNo, 'Incremental', mrCancel, 'Cancelar'],
                    0) of
      mrYes: Incremental := False;
      mrNo: Incremental := True;
      else Exit;
    end;
  end;

  memDownloadLog.Clear;
  pbDownload.Position := 0;
  pbDownload.Style := pbstNormal;
  lblDownloadStatus.Caption := 'Iniciando download...';
  btnDownload.Caption := 'Parar Download';
  
  btnNext.Enabled := False;
  btnBack.Enabled := False;

  FDownloaderThread := TModelDownloaderThread.Create(Self, edtDestDir.Text, SelectedModel, Incremental);
  FDownloaderThread.Start;
end;

procedure TfrmInstaller.OnDownloadProgress(const AFileName: string; AMax, AVal: Int64);
begin
  if AMax > 0 then
  begin
    pbDownload.Max := 100;
    pbDownload.Position := Round((AVal / AMax) * 100);
  end
  else
  begin
    pbDownload.Style := pbstMarquee;
  end;
end;

procedure TfrmInstaller.OnDownloadStatus(const AMsg: string; AIsError: Boolean);
begin
  lblDownloadStatus.Caption := AMsg;
  if AIsError then
    lblDownloadStatus.Font.Color := clRed
  else
    lblDownloadStatus.Font.Color := clDefault;
end;

procedure TfrmInstaller.OnDownloadLog(const AMsg: string);
begin
  memDownloadLog.Lines.Add(AMsg);
end;

procedure TfrmInstaller.OnDownloadFinished(ASuccess: Boolean; const AReport: string);
var
  RepForm: TfrmDownloadReport;
begin
  FDownloaderThread := nil;
  btnDownload.Caption := 'Baixar Modelos Selecionados';
  pbDownload.Style := pbstNormal;
  if ASuccess then
    pbDownload.Position := 100
  else
    pbDownload.Position := 0;
  
  btnBack.Enabled := True;
  FDownloadReport := AReport;

  // Criação da janela de relatório usando a nova classe TfrmDownloadReport
  RepForm := TfrmDownloadReport.Create(Self);
  try
    RepForm.Memo.Lines.Text := AReport;
    RepForm.ShowModal;
  finally
    RepForm.Free;
  end;

  FDownloadAllowedNext := ASuccess;
  UpdateWizardState;
end;

procedure TfrmInstaller.btnSendQuestionClick(Sender: TObject);
begin
  if Assigned(FChatThread) then Exit;

  memResponse.Text := 'Enviando pergunta para a IA, por favor aguarde...';
  btnSendQuestion.Enabled := False;

  FChatThread := TChatTestThread.Create(
    Self,
    TAIProvider(cmbProvider.ItemIndex),
    edtToken.Text,
    edtURL.Text,
    edtCustomModel.Text,
    memQuestion.Text
  );
  FChatThread.Start;
end;

procedure TfrmInstaller.OnChatFinished(ASuccess: Boolean; const AResponse: string);
begin
  FChatThread := nil;
  btnSendQuestion.Enabled := True;
  memResponse.Text := AResponse;
end;

procedure TfrmInstaller.btnStartServerClick(Sender: TObject);
var
  Idx: Integer;
begin
  if Assigned(FChatServerThread) then
  begin
    TChatServerThread(FChatServerThread).StopServer;
    FChatServerThread := nil;
    btnStartServer.Caption := 'Iniciar Servidor';
    lblServerStatus.Caption := 'Aguardando inicialização do servidor...';
    lblServerStatus.Font.Color := clDefault;
    FServerAllowedNext := False;
    UpdateWizardState;
    Exit;
  end;

  Idx := cmbModels.ItemIndex;
  if (Idx < 0) or (Idx >= Length(FModelsList)) then Exit;

  memServerLog.Clear;
  btnStartServer.Caption := 'Parar Servidor';
  FServerAllowedNext := False;
  UpdateWizardState;

  FChatServerThread := TChatServerThread.Create(Self, edtDestDir.Text, FModelsList[Idx].Folder, edtServerPort.Text);
  TChatServerThread(FChatServerThread).Start;
end;

procedure TfrmInstaller.OnServerActive(ASuccess: Boolean; const AMsg: string);
begin
  lblServerStatus.Caption := AMsg;
  FServerAllowedNext := ASuccess;
  
  if ASuccess then
    lblServerStatus.Font.Color := clGreen
  else
    lblServerStatus.Font.Color := clRed;
  
  if not ASuccess and (FChatServerThread <> nil) then
  begin
    // Se a thread terminou e não teve sucesso
    if TChatServerThread(FChatServerThread).Finished then
    begin
      FChatServerThread := nil;
      btnStartServer.Caption := 'Iniciar Servidor';
    end;
  end;
  
  UpdateWizardState;
end;

procedure TfrmInstaller.OnServerLog(const AMsg: string);
begin
  memServerLog.Lines.Add(AMsg);
end;

procedure TfrmInstaller.cmbLanguageChange(Sender: TObject);
begin
  UpdateLanguageUI(cmbLanguage.ItemIndex);
  UpdateWizardState;
end;

procedure TfrmInstaller.UpdateLanguageUI(ALang: Integer);
begin
  // Configuração de RTL (Right-to-Left) para Árabe
  if ALang = 5 then
    Self.BidiMode := bdRightToLeft
  else
    Self.BidiMode := bdLeftToRight;

  case ALang of
    0: { English }
      begin
        lblTitle.Caption := 'INSTALLATION';
        lblStepWelcome.Caption := '1. Welcome';
        lblStepPath.Caption := '2. Destination Folder';
        lblStepModels.Caption := '3. Download Models';
        lblStepServer.Caption := '4. Start Server';
        lblStepTest.Caption := '5. AI Test';
        lblStepFinish.Caption := '6. Finish';

        btnCancel.Caption := 'Cancel';
        btnBack.Caption := '< Back';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Finish'
        else
          btnNext.Caption := 'Next >';

        lblWelcome.Caption := 'Welcome to CAI Neural API';
        lblWelcomeDesc.Caption := 'This assistant will guide you through installation, downloading models, and testing connection.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API is a robust Pascal deep learning library optimized for AVX instructions and OpenCL (GPU) devices.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Highlights:');
        memWelcomeInfo.Lines.Add('- Run real LLMs natively and fast (no Python or CUDA dependencies).');
        memWelcomeInfo.Lines.Add('- Built-in support for Transformers, MoE, RWKV, and xLSTM.');
        memWelcomeInfo.Lines.Add('- Compiles to a single optimized native binary.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Click Next to proceed with the setup.');

        lblDestDir.Caption := 'Select the root folder where the repository is located:';
        lblLazarusDir.Caption := 'Select Lazarus installation folder:';
        btnBrowse.Caption := 'Browse...';

        lblSelectFiles.Caption := 'Select the instruct model and files to download:';
        btnDownload.Caption := 'Download Selected Models';
        lblDownloadStatus.Caption := 'Status: Ready to download.';

        lblServerStatus.Caption := 'Awaiting server startup...';
        btnStartServer.Caption := 'Start Server';

        lblProvider.Caption := 'Provider:';
        lblToken.Caption := 'Token / API Key:';
        lblURL.Caption := 'Endpoint URL:';
        lblCustomModel.Caption := 'Model Name:';
        lblQuestion.Caption := 'Your Question:';
        lblResponse.Caption := 'Test Response:';
        btnSendQuestion.Caption := 'Test Connection / Send Question';

        lblFinishTitle.Caption := 'Installation Finished!';
        lblFinishDesc.Caption := 'CAI Neural API has been configured successfully.' + #13#10#13#10 + 'You selected the path and downloaded all necessary files to run the repository examples.';
      end;
    1: { Português }
      begin
        lblTitle.Caption := 'INSTALAÇÃO';
        lblStepWelcome.Caption := '1. Boas-vindas';
        lblStepPath.Caption := '2. Pasta de Destino';
        lblStepModels.Caption := '3. Download de Modelos';
        lblStepServer.Caption := '4. Iniciar Servidor';
        lblStepTest.Caption := '5. Teste de IA';
        lblStepFinish.Caption := '6. Conclusão';

        btnCancel.Caption := 'Cancelar';
        btnBack.Caption := '< Voltar';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Concluir'
        else
          btnNext.Caption := 'Avançar >';

        lblWelcome.Caption := 'Bem-vindo ao CAI Neural API';
        lblWelcomeDesc.Caption := 'Este assistente irá guiar você através da instalação, download de modelos e configuração de teste.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('O CAI NEURAL API é uma biblioteca Pascal robusta para redes neurais, otimizada para instruções AVX e OpenCL (GPU).');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Destaques:');
        memWelcomeInfo.Lines.Add('- Execute LLMs reais de forma nativa e rápida (sem Python ou CUDA).');
        memWelcomeInfo.Lines.Add('- Suporte embutido para Transformers, MoE, RWKV e xLSTM.');
        memWelcomeInfo.Lines.Add('- Executável nativo único e otimizado.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Clique em Avançar para prosseguir com a configuração.');

        lblDestDir.Caption := 'Selecione a pasta raiz onde o repositório está localizado:';
        lblLazarusDir.Caption := 'Selecione a pasta onde o Lazarus está instalado:';
        btnBrowse.Caption := 'Procurar...';

        lblSelectFiles.Caption := 'Selecione o modelo instruct que deseja baixar e os respectivos arquivos:';
        btnDownload.Caption := 'Baixar Modelos Selecionados';
        lblDownloadStatus.Caption := 'Status: Pronto para baixar.';

        lblServerStatus.Caption := 'Aguardando inicialização do servidor...';
        btnStartServer.Caption := 'Iniciar Servidor';

        lblProvider.Caption := 'Provedor:';
        lblToken.Caption := 'Token / API Key:';
        lblURL.Caption := 'URL Customizada:';
        lblCustomModel.Caption := 'Nome do Modelo:';
        lblQuestion.Caption := 'Sua Pergunta:';
        lblResponse.Caption := 'Resposta do Teste:';
        btnSendQuestion.Caption := 'Testar Conexão / Enviar Pergunta';

        lblFinishTitle.Caption := 'Instalação Concluída!';
        lblFinishDesc.Caption := 'O CAI Neural API foi configurado com sucesso.' + #13#10#13#10 + 'Você selecionou a pasta e baixou os arquivos necessários para rodar os exemplos do repositório.';
      end;
    2: { Français }
      begin
        lblTitle.Caption := 'INSTALLATION';
        lblStepWelcome.Caption := '1. Bienvenue';
        lblStepPath.Caption := '2. Dossier Destination';
        lblStepModels.Caption := '3. Charger Modèles';
        lblStepServer.Caption := '4. Démarrer Serveur';
        lblStepTest.Caption := '5. Test de l''IA';
        lblStepFinish.Caption := '6. Terminer';

        btnCancel.Caption := 'Annuler';
        btnBack.Caption := '< Précédent';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Terminer'
        else
          btnNext.Caption := 'Suivant >';

        lblWelcome.Caption := 'Bienvenue sur CAI Neural API';
        lblWelcomeDesc.Caption := 'Ce magicien vous guidera tout au long de l''installation, du téléchargement des modèles et du test.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API est une bibliothèque Pascal robuste pour le deep learning, optimisée pour AVX et OpenCL.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Points forts :');
        memWelcomeInfo.Lines.Add('- Exécutez de vrais LLM nativement et rapidement (sans dépendance Python ni CUDA).');
        memWelcomeInfo.Lines.Add('- Support intégré pour Transformers, MoE, RWKV et xLSTM.');
        memWelcomeInfo.Lines.Add('- Compile en un seul exécutable natif optimisé.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Cliquez sur Suivant pour continuer.');

        lblDestDir.Caption := 'Sélectionnez le dossier racine où se trouve le dépôt :';
        lblLazarusDir.Caption := 'Sélectionnez le dossier où Lazarus est installé :';
        btnBrowse.Caption := 'Parcourir...';

        lblSelectFiles.Caption := 'Sélectionnez le modèle instruct et les fichiers à télécharger :';
        btnDownload.Caption := 'Télécharger les Modèles Sélectionnés';
        lblDownloadStatus.Caption := 'Statut : Prêt à télécharger.';

        lblServerStatus.Caption := 'En attente du démarrage du serveur...';
        btnStartServer.Caption := 'Démarrer le Serveur';

        lblProvider.Caption := 'Fournisseur :';
        lblToken.Caption := 'Jeton / Clé API :';
        lblURL.Caption := 'URL du point de terminaison :';
        lblCustomModel.Caption := 'Nom du Modèle :';
        lblQuestion.Caption := 'Votre Question :';
        lblResponse.Caption := 'Réponse de Test :';
        btnSendQuestion.Caption := 'Tester la Connexion / Envoyer la Question';

        lblFinishTitle.Caption := 'Installation Terminée !';
        lblFinishDesc.Caption := 'CAI Neural API a été configuré avec succès.' + #13#10#13#10 + 'Vous avez configuré le chemin d''accès et téléchargé tous les fichiers requis.';
      end;
    3: { Deutsch }
      begin
        lblTitle.Caption := 'INSTALLATION';
        lblStepWelcome.Caption := '1. Willkommen';
        lblStepPath.Caption := '2. Zielordner';
        lblStepModels.Caption := '3. Modelle laden';
        lblStepServer.Caption := '4. Server starten';
        lblStepTest.Caption := '5. KI-Test';
        lblStepFinish.Caption := '6. Fertigstellen';

        btnCancel.Caption := 'Abbrechen';
        btnBack.Caption := '< Zurück';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Fertigstellen'
        else
          btnNext.Caption := 'Weiter >';

        lblWelcome.Caption := 'Willkommen bei CAI Neural API';
        lblWelcomeDesc.Caption := 'Dieser Assistent führt Sie durch die Installation, das Herunterladen von Modellen und das Testen.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API ist eine robuste Pascal-Deep-Learning-Bibliothek, optimiert für AVX-Instruktionen und OpenCL.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Highlights:');
        memWelcomeInfo.Lines.Add('- Führen Sie echte LLMs nativ und schnell aus (keine Python- oder CUDA-Abhängigkeiten).');
        memWelcomeInfo.Lines.Add('- Integrierte Unterstützung für Transformers, MoE, RWKV und xLSTM.');
        memWelcomeInfo.Lines.Add('- Kompiliert in eine einzige optimierte native ausführbare Datei.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Klicken Sie auf Weiter, um fortzufahren.');

        lblDestDir.Caption := 'Wählen Sie den Stammordner aus, in dem sich das Repository befindet:';
        lblLazarusDir.Caption := 'Wählen Sie den Ordner aus, in dem Lazarus installiert ist:';
        btnBrowse.Caption := 'Durchsuchen...';

        lblSelectFiles.Caption := 'Wählen Sie das Instruct-Modell und die Dateien zum Herunterladen aus:';
        btnDownload.Caption := 'Ausgewählte Modelle herunterladen';
        lblDownloadStatus.Caption := 'Status: Bereit zum Herunterladen.';

        lblServerStatus.Caption := 'Warten auf Serverstart...';
        btnStartServer.Caption := 'Server starten';

        lblProvider.Caption := 'Anbieter:';
        lblToken.Caption := 'Token / API-Schlüssel:';
        lblURL.Caption := 'Endpunkt-URL:';
        lblCustomModel.Caption := 'Modellname:';
        lblQuestion.Caption := 'Ihre Frage:';
        lblResponse.Caption := 'Testantwort:';
        btnSendQuestion.Caption := 'Verbindung testen / Frage senden';

        lblFinishTitle.Caption := 'Installation abgeschlossen!';
        lblFinishDesc.Caption := 'CAI Neural API wurde erfolgreich konfiguriert.' + #13#10#13#10 + 'Sie haben den Pfad ausgewählt und alle erforderlichen Modelldateien heruntergeladen.';
      end;
    4: { Español }
      begin
        lblTitle.Caption := 'INSTALACIÓN';
        lblStepWelcome.Caption := '1. Bienvenida';
        lblStepPath.Caption := '2. Carpeta Destino';
        lblStepModels.Caption := '3. Descargar Modelos';
        lblStepServer.Caption := '4. Iniciar Servidor';
        lblStepTest.Caption := '5. Prueba de IA';
        lblStepFinish.Caption := '6. Finalizar';

        btnCancel.Caption := 'Cancelar';
        btnBack.Caption := '< Atrás';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Finalizar'
        else
          btnNext.Caption := 'Siguiente >';

        lblWelcome.Caption := 'Bienvenido a CAI Neural API';
        lblWelcomeDesc.Caption := 'Este asistente le guiará a través de la instalación, descarga de modelos y pruebas.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API es una biblioteca Pascal robusta de deep learning, optimizada para instrucciones AVX y OpenCL.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Características destacadas:');
        memWelcomeInfo.Lines.Add('- Ejecución nativa y rápida de LLM reales (sin dependencias de Python ni CUDA).');
        memWelcomeInfo.Lines.Add('- Soporte nativo para Transformers, MoE, RWKV y xLSTM.');
        memWelcomeInfo.Lines.Add('- Se compila en un único binario nativo optimizado.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Haga clic en Siguiente para continuar.');

        lblDestDir.Caption := 'Seleccione la carpeta raíz donde se encuentra el repositorio:';
        lblLazarusDir.Caption := 'Seleccione la carpeta donde está instalado Lazarus:';
        btnBrowse.Caption := 'Buscar...';

        lblSelectFiles.Caption := 'Seleccione el modelo instruct y los archivos para descargar:';
        btnDownload.Caption := 'Descargar Modelos Seleccionados';
        lblDownloadStatus.Caption := 'Estado: Listo para descargar.';

        lblServerStatus.Caption := 'Esperando inicio del servidor...';
        btnStartServer.Caption := 'Iniciar Servidor';

        lblProvider.Caption := 'Proveedor:';
        lblToken.Caption := 'Token / Clave API:';
        lblURL.Caption := 'URL de endpoint:';
        lblCustomModel.Caption := 'Nombre del Modelo:';
        lblQuestion.Caption := 'Su Pregunta:';
        lblResponse.Caption := 'Respuesta de Prueba:';
        btnSendQuestion.Caption := 'Probar Conexión / Enviar Pregunta';

        lblFinishTitle.Caption := '¡Instalación Finalizada!';
        lblFinishDesc.Caption := 'CAI Neural API se ha configurado correctamente.' + #13#10#13#10 + 'Ha seleccionado la ruta y descargado los archivos de modelo necesarios.';
      end;
    5: { Arabic (RTL) }
      begin
        lblTitle.Caption := 'التثبيت';
        lblStepWelcome.Caption := '١. الترحيب';
        lblStepPath.Caption := '٢. مجلد الوجهة';
        lblStepModels.Caption := '٣. تنزيل النماذج';
        lblStepServer.Caption := '٤. تشغيل الخادم';
        lblStepTest.Caption := '٥. اختبار الذكاء الاصطناعي';
        lblStepFinish.Caption := '٦. إنهاء';

        btnCancel.Caption := 'إلغاء';
        btnBack.Caption := '< السابق';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'إنهاء'
        else
          btnNext.Caption := 'التالي >';

        lblWelcome.Caption := 'مرحبًا بك في CAI Neural API';
        lblWelcomeDesc.Caption := 'سيرشدك هذا المعالج خلال عملية التثبيت وتنزيل النماذج واختبار الاتصال.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API عبارة عن مكتبة باسكال قوية للتعلم العميق، وهي محسنة لتعليمات AVX وأجهزة OpenCL.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('أبرز الميزات:');
        memWelcomeInfo.Lines.Add('- تشغيل نماذج لغوية كبيرة حقيقية بشكل أصلي وسريع (بدون بايثون أو CUDA).');
        memWelcomeInfo.Lines.Add('- دعم مدمج لـ Transformers و MoE و RWKV و xLSTM.');
        memWelcomeInfo.Lines.Add('- يتم ترجمتها وتجميعها في ملف تنفيذي أصلي واحد ومحسن.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('انقر فوق التالي للمتابعة.');

        lblDestDir.Caption := 'حدد المجلد الجذر حيث يوجد المستودع:';
        lblLazarusDir.Caption := 'حدد المجلد الذي تم تثبيت Lazarus فيه:';
        btnBrowse.Caption := 'تصفح...';

        lblSelectFiles.Caption := 'حدد نموذج التوجيه والملفات المراد تنزيلها:';
        btnDownload.Caption := 'تنزيل النماذج المحددة';
        lblDownloadStatus.Caption := 'الحالة: جاهز للتنزيل.';

        lblServerStatus.Caption := 'في انتظار تشغيل الخادم...';
        btnStartServer.Caption := 'تشغيل الخادم';

        lblProvider.Caption := 'المزود:';
        lblToken.Caption := 'الرمز / مفتاح API:';
        lblURL.Caption := 'عنوان URL لنقطة النهاية:';
        lblCustomModel.Caption := 'اسم النموذج:';
        lblQuestion.Caption := 'سؤالك:';
        lblResponse.Caption := 'استجابة الاختبار:';
        btnSendQuestion.Caption := 'اختبار الاتصال / إرسال السؤال';

        lblFinishTitle.Caption := 'تم الانتهاء من التثبيت!';
        lblFinishDesc.Caption := 'تم تكوين CAI Neural API بنجاح.' + #13#10#13#10 + 'لقد حددت المسار وقمت بتنزيل جميع ملفات النماذج المطلوبة لتشغيل أمثلة المستودع.';
      end;
    6: { Chinese }
      begin
        lblTitle.Caption := '程序安装';
        lblStepWelcome.Caption := '1. 欢迎';
        lblStepPath.Caption := '2. 目标文件夹';
        lblStepModels.Caption := '3. 下载模型';
        lblStepServer.Caption := '4. 启动服务器';
        lblStepTest.Caption := '5. AI 测试';
        lblStepFinish.Caption := '6. 完成';

        btnCancel.Caption := '取消';
        btnBack.Caption := '< 返回';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := '完成'
        else
          btnNext.Caption := '下一步 >';

        lblWelcome.Caption := '欢迎使用 CAI Neural API';
        lblWelcomeDesc.Caption := '此安装向导将引导您完成安装、下载模型及测试连接。';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API 是一个强大的 Pascal 深度学习库，专为 AVX 指令集和 OpenCL（GPU）设备优化。');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('主要亮点：');
        memWelcomeInfo.Lines.Add('- 原生且快速运行真实 LLM（无 Python 或 CUDA 依赖）。');
        memWelcomeInfo.Lines.Add('- 内置支持 Transformers、MoE、RWKV 和 xLSTM。');
        memWelcomeInfo.Lines.Add('- 编译为单个优化过的原生可执行文件。');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('点击下一步以继续。');

        lblDestDir.Caption := '选择存储库所在的根文件夹：';
        lblLazarusDir.Caption := '选择 Lazarus 的安装目录：';
        btnBrowse.Caption := '浏览...';

        lblSelectFiles.Caption := '选择要下载的 instruct 模型和文件：';
        btnDownload.Caption := '下载选定的模型';
        lblDownloadStatus.Caption := '状态：准备下载。';

        lblServerStatus.Caption := '等待服务器启动...';
        btnStartServer.Caption := '启动服务器';

        lblProvider.Caption := '提供商:';
        lblToken.Caption := '令牌 / API 密钥:';
        lblURL.Caption := '终点 URL:';
        lblCustomModel.Caption := '模型名称:';
        lblQuestion.Caption := '您的问题:';
        lblResponse.Caption := '测试响应:';
        btnSendQuestion.Caption := '测试连接 / 发送问题';

        lblFinishTitle.Caption := '安装已完成！';
        lblFinishDesc.Caption := 'CAI Neural API 已成功配置。' + #13#10#13#10 + '您已选择路径并下载了运行仓库示例所需的所有模型文件。';
      end;
    7: { Japanese }
      begin
        lblTitle.Caption := 'インストール';
        lblStepWelcome.Caption := '1. 歓迎';
        lblStepPath.Caption := '2. 送信先フォルダ';
        lblStepModels.Caption := '3. モデルダウンロード';
        lblStepServer.Caption := '4. サーバー起動';
        lblStepTest.Caption := '5. AIテスト';
        lblStepFinish.Caption := '6. 完了';

        btnCancel.Caption := 'キャンセル';
        btnBack.Caption := '< 戻る';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := '完了'
        else
          btnNext.Caption := '次へ >';

        lblWelcome.Caption := 'CAI Neural API へようこそ';
        lblWelcomeDesc.Caption := 'このアシスタントは、インストール、モデルのダウンロード、接続のテストを案内します。';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API は、AVX命令およびOpenCL（GPU）デバイス用に最適化された、堅牢なPascalディープラーニングライブラリです。');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('主な特徴：');
        memWelcomeInfo.Lines.Add('- 本物のLLMをネイティブかつ高速に実行（PythonやCUDAの依存関係はありません）。');
        memWelcomeInfo.Lines.Add('- Transformers、MoE、RWKV、およびxLSTM of 組み込みサポート。');
        memWelcomeInfo.Lines.Add('- 最適化された単一のネイティブバイナリにコンパイルします。');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('次へをクリックしてセットアップを続行してください。');

        lblDestDir.Caption := 'リポジトリがあるルートフォルダを選択してください：';
        lblLazarusDir.Caption := 'Lazarusがインストールされているフォルダを選択してください：';
        btnBrowse.Caption := '参照...';

        lblSelectFiles.Caption := 'ダウンロードする instruct モデルとファイルを選択してください：';
        btnDownload.Caption := '選択したモデルをダウンロード';
        lblDownloadStatus.Caption := 'ステータス：ダウンロード可能。';

        lblServerStatus.Caption := 'サーバーの起動を待っています...';
        btnStartServer.Caption := 'サーバーを起動';

        lblProvider.Caption := 'プロバイダー:';
        lblToken.Caption := 'トークン / APIキー:';
        lblURL.Caption := 'エンドポイントURL:';
        lblCustomModel.Caption := 'モデル名:';
        lblQuestion.Caption := '質問:';
        lblResponse.Caption := 'テスト応答:';
        btnSendQuestion.Caption := '接続テスト / 質問送信';

        lblFinishTitle.Caption := 'インストールが完了しました！';
        lblFinishDesc.Caption := 'CAI Neural API が正常に設定されました。' + #13#10#13#10 + 'パスの選択と、リポジトリ of サンプル実行に必要なモデルファイルのダウンロードが完了しました。';
      end;
    8: { Russian }
      begin
        lblTitle.Caption := 'УСТАНОВКА';
        lblStepWelcome.Caption := '1. Приветствие';
        lblStepPath.Caption := '2. Папка назначения';
        lblStepModels.Caption := '3. Загрузка моделей';
        lblStepServer.Caption := '4. Запуск сервера';
        lblStepTest.Caption := '5. Тест ИИ';
        lblStepFinish.Caption := '6. Завершение';

        btnCancel.Caption := 'Отмена';
        btnBack.Caption := '< Назад';
        if FCurrentPage = pgcWizard.PageCount - 1 then
          btnNext.Caption := 'Завершить'
        else
          btnNext.Caption := 'Далее >';

        lblWelcome.Caption := 'Добро пожаловать в CAI Neural API';
        lblWelcomeDesc.Caption := 'Этот мастер поможет вам выполнить установку, загрузку моделей и тестирование подключения.';
        memWelcomeInfo.Lines.Clear;
        memWelcomeInfo.Lines.Add('CAI NEURAL API — это надежная библиотека глубокого обучения на Pascal, оптимизированная для инструкций AVX и устройств OpenCL (GPU).');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Основные преимущества:');
        memWelcomeInfo.Lines.Add('- Запуск реальных LLM нативно и быстро (без зависимостей Python или CUDA).');
        memWelcomeInfo.Lines.Add('- Встроенная поддержка Transformers, MoE, RWKV и xLSTM.');
        memWelcomeInfo.Lines.Add('- Компиляция в один оптимизированный нативный исполняемый файл.');
        memWelcomeInfo.Lines.Add('');
        memWelcomeInfo.Lines.Add('Нажмите Далее, чтобы продолжить.');

        lblDestDir.Caption := 'Выберите корневую папку, где расположен репозиторий:';
        lblLazarusDir.Caption := 'Выберите папку, в которую установлен Lazarus:';
        btnBrowse.Caption := 'Обзор...';

        lblSelectFiles.Caption := 'Выберите модель instruct и файлы для загрузки:';
        btnDownload.Caption := 'Скачать выбранные модели';
        lblDownloadStatus.Caption := 'Статус: Готово к загрузке.';

        lblServerStatus.Caption := 'Ожидание запуска сервера...';
        btnStartServer.Caption := 'Запустить Сервер';

        lblProvider.Caption := 'Провайдер:';
        lblToken.Caption := 'Токен / API ключ:';
        lblURL.Caption := 'URL эндпоинта:';
        lblCustomModel.Caption := 'Имя модели:';
        lblQuestion.Caption := 'Ваш вопрос:';
        lblResponse.Caption := 'Ответ теста:';
        btnSendQuestion.Caption := 'Тестировать соединение / Отправить вопрос';

        lblFinishTitle.Caption := 'Установка завершена!';
        lblFinishDesc.Caption := 'Библиотека CAI Neural API успешно настроена.' + #13#10#13#10 + 'Вы выбрали путь и загрузили все файлы моделей, необходимые для запуска примеров из репозитория.';
      end;
  end;
end;

end.
