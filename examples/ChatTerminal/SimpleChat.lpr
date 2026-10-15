program SimpleChat;
(*
SimpleChat: the smallest "ask a chat model, print the reply" program.
'q' is a git clone of Qwen/Qwen2.5-0.5B-Instruct (see the README section
"From your own code"); run it from the folder that holds q.
*)
{$mode objfpc}{$H+}
uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  neuralchat, neuralchatengine;

var
  Options: TChatOptions;
  Engine: TChatEngine;
  ErrorMsg: string;
  History: TChatMessages;
begin
  Options := DefaultChatOptions();          // int8 weights, model's sampling defaults
  Options.ModelDir := 'q';                  // git clone of Qwen/Qwen2.5-0.5B-Instruct
  Options.CtxLen := 2048;                   // keeps the KV cache small
  Engine := TChatEngine.Create();
  try
    if Engine.LoadModel(Options, ErrorMsg) then
    begin
      History := [ChatMessage('user', 'Name three colours.')];
      WriteLn(Engine.ChatReply(History, Engine.Opt));
    end
    else
      WriteLn('Error: ', ErrorMsg);
  finally
    Engine.Free;
  end;
end.
