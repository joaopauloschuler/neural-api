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
    Session.SpeakTextToWav(                       // a long text works too
      'A small red fox lived at the edge of a quiet forest. ' +
      'Every morning she ran down to the river to drink. ' +
      'One cold winter day she counted 3 rabbits in the snow. ' +
      'They looked at her, and she looked at them. ' +
      'Then the fox turned around and went home to sleep.', 'story.wav');
  finally
    Session.Free;
  end;
end.
