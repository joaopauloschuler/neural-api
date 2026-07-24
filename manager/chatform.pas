unit chatform;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, StdCtrls,
  LCLType, chatgpt, aibase;

type
  { TfrmChat }

  TfrmChat = class(TForm)
    memChat: TMemo;
    pnlInput: TPanel;
    pnlRightButtons: TPanel;
    memQuestion: TMemo;
    btnSend: TButton;
    btnStop: TButton;
    btnAttach: TButton;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure btnSendClick(Sender: TObject);
    procedure btnStopClick(Sender: TObject);
    procedure btnAttachClick(Sender: TObject);
    procedure memQuestionKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
    procedure FormShow(Sender: TObject);
  private
    FPort: string;
    FModel: string;
    FLanguageIdx: Integer;
    procedure SetUIState(ABusy: Boolean);
    procedure UpdateUIStrings;
  public
    procedure SetConfig(const APort, AModel: string; ALangIdx: Integer);
  end;

var
  frmChat: TfrmChat;

implementation

{$R *.lfm}

{ TfrmChat }

procedure TfrmChat.FormCreate(Sender: TObject);
begin
  Self.DoubleBuffered := True;
  FPort := '8095';
  FModel := '';
  FLanguageIdx := 1;
  UpdateUIStrings;
end;

procedure TfrmChat.FormDestroy(Sender: TObject);
begin
end;

procedure TfrmChat.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  SetUIState(False);
end;

procedure TfrmChat.SetConfig(const APort, AModel: string; ALangIdx: Integer);
begin
  FPort := APort;
  FModel := AModel;
  FLanguageIdx := ALangIdx;
  UpdateUIStrings;
end;

procedure TfrmChat.FormShow(Sender: TObject);
begin
  memQuestion.SetFocus;
end;

procedure TfrmChat.SetUIState(ABusy: Boolean);
begin
  btnSend.Enabled := not ABusy;
  btnStop.Enabled := False;
  btnAttach.Enabled := not ABusy;
  memQuestion.ReadOnly := ABusy;
  if ABusy then
    Screen.Cursor := crHourGlass
  else
    Screen.Cursor := crDefault;
end;

procedure TfrmChat.btnSendClick(Sender: TObject);
var
  QText: string;
  FChatGPT: TCHATGPT;
  Success: Boolean;
  ResponseText: string;
begin
  QText := Trim(memQuestion.Text);
  if QText = '' then Exit;

  memChat.Lines.Add('>>> Você:');
  memChat.Lines.Add(QText);
  memQuestion.Clear;

  SetUIState(True);
  Application.ProcessMessages;

  FChatGPT := TCHATGPT.Create(nil);
  try
    FChatGPT.Provider := AIP_LOCAL;
    FChatGPT.LocalIP := 'http://127.0.0.1:' + FPort;
    FChatGPT.CustomModel := FModel;
    FChatGPT.TOKEN := 'local'; // chave dummy para API local

    Success := FChatGPT.SendQuestion(QText);
    ResponseText := FChatGPT.Response;

    if Success then
      memChat.Lines.Add('>>> IA: ' + ResponseText)
    else
      memChat.Lines.Add('>>> Erro: ' + ResponseText);
    memChat.Lines.Add('');
  except
    on E: Exception do
    begin
      memChat.Lines.Add('>>> Erro: ' + E.Message);
      memChat.Lines.Add('');
    end;
  end;
  FChatGPT.Free;

  SetUIState(False);
  memQuestion.SetFocus;
end;

procedure TfrmChat.btnStopClick(Sender: TObject);
begin
  // No modo síncrono não há requisição em segundo plano para cancelar
end;

procedure TfrmChat.btnAttachClick(Sender: TObject);
var
  OpenDlg: TOpenDialog;
  Ext: string;
  SL: TStringList;
begin
  OpenDlg := TOpenDialog.Create(Self);
  try
    OpenDlg.Title := 'Selecionar Foto ou Documento';
    OpenDlg.Filter := 'Todos os arquivos (*.*)|*.*|Imagens (*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif)|*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif|Documentos (*.txt;*.md;*.pas;*.py;*.json;*.csv;*.log;*.pdf;*.doc)|*.txt;*.md;*.pas;*.py;*.json;*.csv;*.log;*.pdf;*.doc';
    if OpenDlg.Execute then
    begin
      Ext := LowerCase(ExtractFileExt(OpenDlg.FileName));
      if (Ext = '.png') or (Ext = '.jpg') or (Ext = '.jpeg') or (Ext = '.bmp') or (Ext = '.webp') or (Ext = '.gif') then
      begin
        // Imagem/Foto
        if memQuestion.Text <> '' then
          memQuestion.Lines.Add('');
        memQuestion.Lines.Add('[Foto/Imagem Anexada: ' + ExtractFileName(OpenDlg.FileName) + ' (' + OpenDlg.FileName + ')]');
      end
      else
      begin
        // Documento ou Arquivo de Texto
        try
          SL := TStringList.Create;
          try
            SL.LoadFromFile(OpenDlg.FileName);
            if memQuestion.Text <> '' then
              memQuestion.Lines.Add('');
            memQuestion.Lines.Add('--- Documento Anexado: ' + ExtractFileName(OpenDlg.FileName) + ' ---');
            memQuestion.Lines.Add(SL.Text);
            memQuestion.Lines.Add('--- Fim do Documento ---');
          finally
            SL.Free;
          end;
        except
          on E: Exception do
          begin
            if memQuestion.Text <> '' then
              memQuestion.Lines.Add('');
            memQuestion.Lines.Add('[Documento Anexado: ' + ExtractFileName(OpenDlg.FileName) + ' - Erro ao ler conteúdo: ' + E.Message + ']');
          end;
        end;
      end;
    end;
  finally
    OpenDlg.Free;
  end;
end;

procedure TfrmChat.memQuestionKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  if (Key = VK_RETURN) and (ssCtrl in Shift) then
  begin
    Key := 0;
    btnSendClick(nil);
  end;
end;

procedure TfrmChat.UpdateUIStrings;
begin
  case FLanguageIdx of
    0: { English }
      begin
        Caption := 'Local AI Chat';
        btnSend.Caption := 'Send';
        btnStop.Caption := 'Stop';
        btnAttach.Caption := 'Attach';
      end;
    1: { Portugues }
      begin
        Caption := 'Chat de IA Local';
        btnSend.Caption := 'Enviar';
        btnStop.Caption := 'Parar';
        btnAttach.Caption := 'Anexar';
      end;
    2: { Francais }
      begin
        Caption := 'Chat IA Local';
        btnSend.Caption := 'Envoyer';
        btnStop.Caption := 'Arrêter';
        btnAttach.Caption := 'Joindre';
      end;
    3: { Deutsch }
      begin
        Caption := 'Lokaler KI-Chat';
        btnSend.Caption := 'Senden';
        btnStop.Caption := 'Stopp';
        btnAttach.Caption := 'Anhängen';
      end;
    4: { Espanol }
      begin
        Caption := 'Chat de IA Local';
        btnSend.Caption := 'Enviar';
        btnStop.Caption := 'Detener';
        btnAttach.Caption := 'Adjuntar';
      end;
    5: { Arabic }
      begin
        Caption := 'دردشة ذكاء اصطناعي محلي';
        btnSend.Caption := 'إرسال';
        btnStop.Caption := 'إيقاف';
        btnAttach.Caption := 'إرفاق';
      end;
    6: { Chinese }
      begin
        Caption := '本地 AI 聊天';
        btnSend.Caption := '发送';
        btnStop.Caption := '停止';
        btnAttach.Caption := '附件';
      end;
    7: { Japanese }
      begin
        Caption := 'ローカルAIチャット';
        btnSend.Caption := '送信';
        btnStop.Caption := '停止';
        btnAttach.Caption := '添付';
      end;
    8: { Russian }
      begin
        Caption := 'Локальный ИИ Чат';
        btnSend.Caption := 'Отправить';
        btnStop.Caption := 'Стоп';
        btnAttach.Caption := 'Прикрепить';
      end;
  end;
end;

end.
