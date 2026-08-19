unit chatgpt;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, LazUTF8, fpjson, jsonparser,
  fphttpclient, opensslsockets, LResources, aibase;

const
  CHATGPT_LIB_VERSION = '1.6';

type
  TVersionChat = (
    VCT_GPT35TURBO,
    VCT_GPT40,
    VCT_GPT40_TURBO,
    VCT_GPT4o,
    VCT_GPT4O_MINI,
    VCT_GPTo3_mini,
    VCT_GPTo1,
    VCT_GPTo1_mini,
    VCT_GPTo1_preview,
    VCT_GPT41,
    VCT_GPT41_MINI,
    VCT_GPT5,

    // Modelos locais / Ollama (Totalmente Gratuitos)
    VCT_LLAMA32_3B,
    VCT_QWEN25_15B,
    VCT_DEEPSEEK_R1_15B,
    VCT_DEEPSEEK_R1_8B,
    VCT_DEEPSEEK_R1_14B,
    VCT_DEEPSEEK_R1_70B,

    // Gemini (Google) - Possuem cotas de uso gratuitas
    VCT_GEMINI_15_FLASH,
    VCT_GEMINI_15_PRO,
    VCT_GEMINI_20_FLASH,
    VCT_GEMINI_25_FLASH,
    VCT_GEMINI_25_PRO,

    // Anthropic Claude
    VCT_CLAUDE_35_SONNET,
    VCT_CLAUDE_35_HAIKU,
    VCT_CLAUDE_3_OPUS,

    // Modelos Gratuitos via OpenRouter
    VCT_OPENROUTER_LLAMA3_8B_FREE,
    VCT_OPENROUTER_GEMMA2_9B_FREE,
    VCT_OPENROUTER_DEEPSEEK_R1_FREE,
    VCT_OPENROUTER_LLAMA32_3B_FREE,

    // Modelos locais DeepSeek R1 específicos do usuário
    VCT_DEEPSEEK_R1_1_5b,
    VCT_DEEPSEEK_R1_7b,

    VCT_CUSTOM
  );

  TAIProvider = (
    AIP_OPENAI,      // 0
    AIP_OPENROUTER,  // 1
    AIP_CEREBRAS,    // 2
    AIP_LOCAL,       // 3 - llama.cpp / Ollama local
    AIP_GEMINI,      // 4 - Google Gemini
    AIP_CLAUDE       // 5 - Anthropic Claude
  );

  { TCHATGPT }

  TCHATGPT = class(TAIBaseComponent)
  private
    FToken           : WideString;
    FQuestion        : WideString;
    FResponse        : WideString;
    FDev             : WideString;
    FTipoChat        : TVersionChat;
    FProvider        : TAIProvider;
    FParams          : TStrings;
    FCustomModel     : WideString;
    FOpenRouterTitle : WideString;
    FOpenRouterSite  : WideString;
    FLastJSON        : WideString;
    FMaxTokens       : Integer;
    FLocalIP         : WideString;
    FLastURL         : WideString;
    FURL             : WideString;

    function RequestJson(const LURL, token, ASK: WideString): WideString;
    function PegaMensagem(const JSON: WideString): WideString;
    function GetEndpoint: WideString;
    function GetModelName: WideString;
    function MontaURLChatLocal(const AServidor: WideString): WideString;
    procedure AddProviderHeaders(AHTTP: TFPHttpClient);
    function GetDev: WideString;
    procedure SetDev(const AValue: WideString);
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    function SendQuestion(ASK: WideString): Boolean;
    function TipoModelo: WideString;
    function ProviderName: WideString;
    function VersaoBiblioteca: WideString;
  published
    property TOKEN: WideString read FToken write FToken;
    property Question: WideString read FQuestion;
    property Response: WideString read FResponse write FResponse;
    property Dev: WideString read GetDev write SetDev;
    property TipoChat: TVersionChat read FTipoChat write FTipoChat;
    property Provider: TAIProvider read FProvider write FProvider;
    property CustomModel: WideString read FCustomModel write FCustomModel;
    property LocalIP: WideString read FLocalIP write FLocalIP;
    property MaxTokens: Integer read FMaxTokens write FMaxTokens;
    property URL: WideString read FURL write FURL;

    // Opcionais para OpenRouter
    property OpenRouterTitle: WideString read FOpenRouterTitle write FOpenRouterTitle;
    property OpenRouterSite: WideString read FOpenRouterSite write FOpenRouterSite;

    property LastJSON: WideString read FLastJSON;
    property LastURL: WideString read FLastURL;
  end;

procedure Register;

implementation

function JsonEscape(const S: WideString): WideString;
var
  R: WideString;
begin
  R := StringReplace(S, '\', '\\', [rfReplaceAll]);
  R := StringReplace(R, '"', '\"', [rfReplaceAll]);
  R := StringReplace(R, #13#10, '\n', [rfReplaceAll]);
  R := StringReplace(R, #10, '\n', [rfReplaceAll]);
  R := StringReplace(R, #13, '\n', [rfReplaceAll]);
  Result := R;
end;

function TCHATGPT.MontaURLChatLocal(const AServidor: WideString): WideString;
var
  S: WideString;
begin
  S := Trim(AServidor);

  if S = '' then
    S := 'http://localhost:11434';

  if Copy(S, Length(S), 1) = '/' then
    Delete(S, Length(S), 1);

  Result := S + '/v1/chat/completions';
end;

function TCHATGPT.PegaMensagem(const JSON: WideString): WideString;
var
  CleanJSON: WideString;
  Data: TJSONData;
  JsonObject, MessageObject: TJSONObject;
  ChoicesArray: TJSONArray;
  ContentData: TJSONData;
  Parser: TJSONParser;
begin
  CleanJSON := StringReplace(JSON, '#$0A', '', [rfReplaceAll]);
  Result := '';

  if FProvider = AIP_CLAUDE then
  begin
    Parser := TJSONParser.Create(CleanJSON);
    try
      try
        Data := Parser.Parse;
        try
          if Data.JSONType = jtObject then
          begin
            JsonObject := TJSONObject(Data);
            if JsonObject.Find('content', ChoicesArray) then
            begin
              if (ChoicesArray <> nil) and (ChoicesArray.Count > 0) then
              begin
                if ChoicesArray.Items[0].JSONType = jtObject then
                begin
                  ContentData := ChoicesArray.Objects[0].Find('text');
                  if (ContentData <> nil) and (ContentData.JSONType = jtString) then
                    Result := ContentData.AsString;
                end;
              end;
            end;
          end;
        finally
          Data.Free;
        end;
      except
        Result := '';
      end;
    finally
      Parser.Free;
    end;
    Exit;
  end;

  if FProvider = AIP_GEMINI then
  begin
    Parser := TJSONParser.Create(CleanJSON);
    try
      try
        Data := Parser.Parse;
        try
          if Data.JSONType = jtObject then
          begin
            JsonObject := TJSONObject(Data);
            if JsonObject.Find('candidates', ChoicesArray) then
            begin
              if (ChoicesArray <> nil) and (ChoicesArray.Count > 0) then
              begin
                if ChoicesArray.Items[0].JSONType = jtObject then
                begin
                  MessageObject := ChoicesArray.Objects[0].Find('content') as TJSONObject;
                  if MessageObject <> nil then
                  begin
                    if MessageObject.Find('parts', ChoicesArray) then
                    begin
                      if (ChoicesArray <> nil) and (ChoicesArray.Count > 0) then
                      begin
                        if ChoicesArray.Items[0].JSONType = jtObject then
                        begin
                          ContentData := ChoicesArray.Objects[0].Find('text');
                          if (ContentData <> nil) and (ContentData.JSONType = jtString) then
                            Result := ContentData.AsString;
                        end;
                      end;
                    end;
                  end;
                end;
              end;
            end;
          end;
        finally
          Data.Free;
        end;
      except
        Result := '';
      end;
    finally
      Parser.Free;
    end;
    Exit;
  end;

  Parser := TJSONParser.Create(CleanJSON);
  try
    try
      Data := Parser.Parse;
      try
        if Data.JSONType = jtObject then
        begin
          JsonObject := TJSONObject(Data);

          if JsonObject.Find('choices', ChoicesArray) then
          begin
            if (ChoicesArray <> nil) and (ChoicesArray.Count > 0) then
            begin
              if ChoicesArray.Items[0].JSONType = jtObject then
              begin
                MessageObject := ChoicesArray.Objects[0].FindPath('message') as TJSONObject;
                if MessageObject <> nil then
                begin
                  ContentData := MessageObject.Find('content');
                  if (ContentData <> nil) and (ContentData.JSONType = jtString) then
                    Result := ContentData.AsString;
                end;
              end;
            end;
          end;
        end;
      finally
        Data.Free;
      end;
    except
      // Falha silenciosa no parser retorna a mensagem bruta
      Result := '';
    end;
  finally
    Parser.Free;
  end;
end;

function TCHATGPT.GetEndpoint: WideString;
var
  LURL: WideString;
begin
  if Trim(FURL) <> '' then
  begin
    LURL := Trim(FURL);
    if (FProvider = AIP_LOCAL) or (Pos('127.0.0.1', LURL) > 0) or (Pos('localhost', LURL) > 0) then
    begin
      if (Pos('/v1/chat/completions', LURL) = 0) and (Pos('/v1/completions', LURL) = 0) then
      begin
        if (Length(LURL) > 0) and (LURL[Length(LURL)] = '/') then
          LURL := LURL + 'v1/chat/completions'
        else
          LURL := LURL + '/v1/chat/completions';
      end;
    end;
    Result := LURL;
    Exit;
  end;

  case FProvider of
    AIP_OPENAI:
      Result := 'https://api.openai.com/v1/chat/completions';

    AIP_OPENROUTER:
      Result := 'https://openrouter.ai/api/v1/chat/completions';

    AIP_CEREBRAS:
      Result := 'https://api.cerebras.ai/v1/chat/completions';

    AIP_GEMINI:
      Result := 'https://generativelanguage.googleapis.com/v1beta/models/' + GetModelName + ':generateContent?key=' + FToken;

    AIP_CLAUDE:
      Result := 'https://api.anthropic.com/v1/messages';

    AIP_LOCAL:
      Result := MontaURLChatLocal(FLocalIP);
  else
    Result := 'https://api.openai.com/v1/chat/completions';
  end;
end;

function TCHATGPT.GetModelName: WideString;
begin
  // Se informou modelo customizado, respeita sempre.
  if Trim(FCustomModel) <> '' then
    Exit(Trim(FCustomModel));

  // Local / Ollama — mapeia enums específicos
  if FProvider = AIP_LOCAL then
  begin
    case FTipoChat of
      VCT_LLAMA32_3B:       Result := 'llama3.2:3b';
      VCT_QWEN25_15B:       Result := 'qwen2.5:1.5b';
      VCT_DEEPSEEK_R1_15B:  Result := 'deepseek-r1:1.5b';
      VCT_DEEPSEEK_R1_8B:   Result := 'deepseek-r1:8b';
      VCT_DEEPSEEK_R1_14B:  Result := 'deepseek-r1:14b';
      VCT_DEEPSEEK_R1_70B:  Result := 'deepseek-r1:70b';
      VCT_DEEPSEEK_R1_1_5b: Result := 'deepseek_r1:1_5b';
      VCT_DEEPSEEK_R1_7b:   Result := 'deepseek_r1:7b';
    else
      Result := 'llama3.2:3b';
    end;
    Exit;
  end;

  // Cerebras
  if FProvider = AIP_CEREBRAS then
  begin
    Exit('qwen-3-235b-a22b-instruct-2507');
  end;

  // OpenRouter
  if FProvider = AIP_OPENROUTER then
  begin
    case FTipoChat of
      VCT_OPENROUTER_LLAMA3_8B_FREE:   Result := 'meta-llama/llama-3-8b-instruct:free';
      VCT_OPENROUTER_GEMMA2_9B_FREE:   Result := 'google/gemma-2-9b-it:free';
      VCT_OPENROUTER_DEEPSEEK_R1_FREE:  Result := 'deepseek/deepseek-r1:free';
      VCT_OPENROUTER_LLAMA32_3B_FREE:  Result := 'meta-llama/llama-3.2-3b-instruct:free';
    else
      Result := 'google/gemma-2-9b-it:free';
    end;
    Exit;
  end;

  // Gemini
  if FProvider = AIP_GEMINI then
  begin
    case FTipoChat of
      VCT_GEMINI_15_FLASH: Result := 'gemini-2.5-flash'; // Fallback para modelo legado descontinuado pela Google
      VCT_GEMINI_15_PRO:   Result := 'gemini-2.5-pro';   // Fallback para modelo legado descontinuado pela Google
      VCT_GEMINI_20_FLASH: Result := 'gemini-2.0-flash';
      VCT_GEMINI_25_FLASH: Result := 'gemini-2.5-flash';
      VCT_GEMINI_25_PRO:   Result := 'gemini-2.5-pro';
    else
      Result := 'gemini-2.5-flash';
    end;
    Exit;
  end;

  // Anthropic Claude
  if FProvider = AIP_CLAUDE then
  begin
    case FTipoChat of
      VCT_CLAUDE_35_SONNET: Result := 'claude-3-5-sonnet-20241022';
      VCT_CLAUDE_35_HAIKU:  Result := 'claude-3-5-haiku-20241022';
      VCT_CLAUDE_3_OPUS:    Result := 'claude-3-opus-20240229';
    else
      Result := 'claude-3-5-sonnet-20241022';
    end;
    Exit;
  end;

  // OpenAI
  case FTipoChat of
    VCT_GPT35TURBO:    Result := 'gpt-3.5-turbo';
    VCT_GPT40:         Result := 'gpt-4';
    VCT_GPT40_TURBO:   Result := 'gpt-4-turbo';
    VCT_GPT4o:         Result := 'gpt-4o';
    VCT_GPT4O_MINI:    Result := 'gpt-4o-mini';
    VCT_GPTo3_mini:    Result := 'o3-mini';
    VCT_GPTo1:         Result := 'o1';
    VCT_GPTo1_mini:    Result := 'o1-mini';
    VCT_GPTo1_preview: Result := 'o1-preview';
    VCT_GPT41:         Result := 'gpt-4.1';
    VCT_GPT41_MINI:    Result := 'gpt-4.1-mini';
    VCT_GPT5:          Result := 'gpt-5';
    VCT_CUSTOM:        Result := Trim(FCustomModel);
  else
    Result := 'gpt-4o';
  end;
end;

procedure TCHATGPT.AddProviderHeaders(AHTTP: TFPHttpClient);
begin
  if AHTTP = nil then
    Exit;

  AHTTP.AddHeader('Content-Type', 'application/json');
  AHTTP.AddHeader('Accept', 'application/json');

  // Local / llama.cpp / Gemini não necessitam de Bearer Token por padrão
  if (FProvider = AIP_LOCAL) or (FProvider = AIP_GEMINI) then
    Exit;

  if FProvider = AIP_CLAUDE then
  begin
    AHTTP.AddHeader('x-api-key', FToken);
    AHTTP.AddHeader('anthropic-version', '2023-06-01');
    Exit;
  end;

  if Trim(FToken) <> '' then
    AHTTP.AddHeader('Authorization', 'Bearer ' + FToken);

  if FProvider = AIP_OPENROUTER then
  begin
    if Trim(FOpenRouterSite) <> '' then
      AHTTP.AddHeader('HTTP-Referer', FOpenRouterSite);

    if Trim(FOpenRouterTitle) <> '' then
      AHTTP.AddHeader('X-OpenRouter-Title', FOpenRouterTitle);
  end;
end;

function TCHATGPT.RequestJson(const LURL, token, ASK: WideString): WideString;
var
  ClienteHTTP: TFPHttpClient;
  BodyStream: TStringStream;
  root, mSys, mUser: TJSONObject;
  msgs: TJSONArray;
  payload: UTF8String;
  GeminiContentArr, GeminiPartsArr: TJSONArray;
  GeminiContentObj, GeminiPartObj: TJSONObject;
begin
  if FProvider = AIP_CLAUDE then
  begin
    root := TJSONObject.Create;
    try
      root.Add('model', GetModelName);
      if Trim(FDev) <> '' then
        root.Add('system', FDev);

      msgs := TJSONArray.Create;
      root.Add('messages', msgs);

      mUser := TJSONObject.Create;
      mUser.Add('role', 'user');
      mUser.Add('content', ASK);
      msgs.Add(mUser);

      if FMaxTokens > 0 then
        root.Add('max_tokens', FMaxTokens)
      else
        root.Add('max_tokens', 4096);

      payload := UTF8Encode(root.AsJSON);
    finally
      root.Free;
    end;
  end
  else if FProvider = AIP_GEMINI then
  begin
    root := TJSONObject.Create;
    try
      if Trim(FDev) <> '' then
      begin
        mSys := TJSONObject.Create;
        msgs := TJSONArray.Create;
        mUser := TJSONObject.Create;
        mUser.Add('text', FDev);
        msgs.Add(mUser);
        mSys.Add('parts', msgs);
        root.Add('systemInstruction', mSys);
      end;

      GeminiContentArr := TJSONArray.Create;
      GeminiContentObj := TJSONObject.Create;
      GeminiPartsArr := TJSONArray.Create;
      GeminiPartObj := TJSONObject.Create;
      
      GeminiPartObj.Add('text', ASK);
      GeminiPartsArr.Add(GeminiPartObj);
      GeminiContentObj.Add('parts', GeminiPartsArr);
      GeminiContentArr.Add(GeminiContentObj);
      
      root.Add('contents', GeminiContentArr);

      payload := UTF8Encode(root.AsJSON);
    finally
      root.Free;
    end;
  end
  else
  begin
    root := TJSONObject.Create;
    try
      root.Add('model', GetModelName);

      msgs := TJSONArray.Create;
      root.Add('messages', msgs);

      if Trim(FDev) <> '' then
      begin
        mSys := TJSONObject.Create;
        mSys.Add('role', 'system');
        mSys.Add('content', FDev);
        msgs.Add(mSys);
      end;

      mUser := TJSONObject.Create;
      mUser.Add('role', 'user');
      mUser.Add('content', ASK);
      msgs.Add(mUser);

      root.Add('temperature', 0.7);
      if FMaxTokens > 0 then
        root.Add('max_tokens', FMaxTokens);

      payload := UTF8Encode(root.AsJSON);
    finally
      root.Free;
    end;
  end;

  ClienteHTTP := TFPHttpClient.Create(nil);
  BodyStream := TStringStream.Create(payload);
  try
    AddProviderHeaders(ClienteHTTP);

    ClienteHTTP.AllowRedirect := True;
    ClienteHTTP.KeepConnection := True;

    if FProvider = AIP_LOCAL then
    begin
      ClienteHTTP.IOTimeout := 1500000;
      ClienteHTTP.ConnectTimeout := 1500000;
    end
    else
    begin
      ClienteHTTP.IOTimeout := 360000;
      ClienteHTTP.ConnectTimeout := 360000;
    end;

    ClienteHTTP.RequestBody := BodyStream;

    try
      Result := ClienteHTTP.Post(UTF8Encode(LURL));
    except
      on E: Exception do
        Result := Format('{"error":{"message":"%s"}}',
          [StringReplace(E.Message, '"', '\"', [rfReplaceAll])]);
    end;
  finally
    BodyStream.Free;
    ClienteHTTP.Free;
  end;
end;

function TCHATGPT.GetDev: WideString;
begin
  if FPrompt <> '' then
    Result := FPrompt
  else
    Result := FDev;
end;

procedure TCHATGPT.SetDev(const AValue: WideString);
begin
  FDev := AValue;
  FPrompt := AValue;
end;

function TCHATGPT.SendQuestion(ASK: WideString): Boolean;
var
  LURL, AUX: WideString;
  ErrorParser: TJSONParser;
  ErrorData: TJSONData;
  ErrorObj: TJSONObject;
  HasError: Boolean;
begin
  Result := False;
  ClearError;
  FQuestion := ASK;

  try
    try
      LURL := GetEndpoint;
      FLastURL := LURL;
    except
      on E: Exception do
      begin
        FResponse := Format('{"error":{"message":"%s"}}',
          [StringReplace(E.Message, '"', '\"', [rfReplaceAll])]);
        FLastJSON := FResponse;
        SetError(E.Message);
        Exit(False);
      end;
    end;

    AUX := RequestJson(LURL, FToken, ASK);
    FLastJSON := AUX;

    // Verifica erro no JSON retornado por parse estruturado
    HasError := False;
    ErrorParser := TJSONParser.Create(AUX);
    try
      try
        ErrorData := ErrorParser.Parse;
        try
          if (ErrorData.JSONType = jtObject) then
          begin
            ErrorObj := TJSONObject(ErrorData);
            if ErrorObj.IndexOfName('error') >= 0 then
            begin
              HasError := True;
              if ErrorObj.Objects['error'] <> nil then
              begin
                if ErrorObj.Objects['error'].IndexOfName('message') >= 0 then
                  SetError(ErrorObj.Objects['error'].Strings['message'])
                else
                  SetError('Erro retornado pela API.');
              end
              else
                SetError('Erro retornado pela API.');
            end;
          end;
        finally
          ErrorData.Free;
        end;
      except
        // Se não conseguir parsear, não é erro de API
        HasError := False;
      end;
    finally
      ErrorParser.Free;
    end;

    if HasError then
    begin
      FResponse := AUX;
      FLastResult := FResponse;
      Exit(False);
    end;

    try
      FResponse := PegaMensagem(AUX);
      Result := (Trim(FResponse) <> '');

      if not Result then
      begin
        FResponse := AUX;
        SetError('Resposta vazia da API.');
      end
      else
      begin
        FLastResult := FResponse;
        FLastSuccess := True;
      end;
    except
      on E: Exception do
      begin
        FResponse := AUX;
        SetError(E.Message);
        Result := False;
      end;
    end;
  except
    on E: Exception do
    begin
      SetError(E.Message);
      Result := False;
    end;
  end;
end;

constructor TCHATGPT.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FProvider := AIP_OPENAI;
  FTipoChat := VCT_GPT4o;

  FDev := 'Você é um assistente.';
  FPrompt := FDev;
  FParams := TStringList.Create;
  FCustomModel := '';
  FOpenRouterTitle := '';
  FOpenRouterSite := '';
  FLastJSON := '';
  FLocalIP := 'http://localhost:11434';
  FMaxTokens := 4096;
  FLastURL := '';
  FURL := '';
end;

destructor TCHATGPT.Destroy;
begin
  FParams.Free;
  inherited;
end;

function TCHATGPT.TipoModelo: WideString;
begin
  Result := '"' + GetModelName + '"';
end;

function TCHATGPT.ProviderName: WideString;
begin
  case FProvider of
    AIP_OPENAI:     Result := 'OpenAI';
    AIP_OPENROUTER: Result := 'OpenRouter';
    AIP_CEREBRAS:   Result := 'Cerebras';
    AIP_LOCAL:      Result := 'Local';
    AIP_GEMINI:     Result := 'Gemini';
    AIP_CLAUDE:     Result := 'Claude';
  else
    Result := 'OpenAI';
  end;
end;

function TCHATGPT.VersaoBiblioteca: WideString;
begin
  Result := CHATGPT_LIB_VERSION;
end;

procedure Register;
begin
  RegisterComponents('AI Core', [TCHATGPT]);
end;

initialization
  {$I chatgpt_icon.lrs}

end.
