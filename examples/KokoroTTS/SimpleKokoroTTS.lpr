program SimpleKokoroTTS;
(*
SimpleKokoroTTS: the smallest English text -> WAV program for Kokoro-82M.
'k' and 'misaki' are the two git clones of the README Quick start; run it
from the folder that holds them (24 kHz mono WAVs).
*)
{$mode objfpc}{$H+}
uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  neuralkokorosession;

var
  Options: TKokoroSessionOptions;
  Session: TKokoroSession;
begin
  Options := DefaultKokoroSessionOptions();      // voice af_heart, speed 1, CPU
  Options.ModelDir := 'k';                        // git clone of hexgrad/Kokoro-82M
  Options.DictionaryDir := 'misaki/misaki/data';  // git clone of hexgrad/misaki
  Session := TKokoroSession.Create(Options);      // loads the model and voice once
  try
    Session.SpeakTextToWav('A red fox in the snow.', 'fox.wav');
    Session.SpeakTextToWav('It jumped over the fence.', 'fence.wav');
  finally
    Session.Free;
  end;
end.
