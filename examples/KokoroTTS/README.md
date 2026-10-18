# KokoroTTS — Kokoro-82M text-to-speech

## Quick start

```
git clone https://huggingface.co/hexgrad/Kokoro-82M k
git clone https://github.com/hexgrad/misaki
lazbuild -B neural-api/examples/KokoroTTS/KokoroTTS.lpi
neural-api/bin/x86_64-linux/bin/KokoroTTS ./k --dict-dir misaki/misaki/data -p "A red fox in the snow"
```

Run these in the folder that holds `neural-api`. The model's `.pth` and
voice files are stored with Git LFS, so `git lfs` must be installed before
the first clone. The voice goes to `kokoro_001.wav` (24 kHz, mono).

## From your own code

```pascal
program SimpleKokoroTTS;
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
```

This is `SimpleKokoroTTS.lpr` in this folder. Build it with
`lazbuild -B neural-api/examples/KokoroTTS/SimpleKokoroTTS.lpi` and run
`neural-api/bin/x86_64-linux/bin/SimpleKokoroTTS` from the folder that holds
`k` and `misaki`; it writes `fox.wav` (one sentence) and `story.wav` (five
sentences, about 16 s of audio).

* `TKokoroSession.Create` loads the model and the voice once.
* `SpeakTextToWav` can be called as many times as you like.
* Set `Options.VoiceName` (e.g. `bf_emma`), `Options.Speed` or
  `Options.Gpu := true` (OpenCL build; stage A only, see How it works)
  before `Create`.
* `SpeakTextToWav` returns a `TKokoroLineResult`: timing, sample count and a
  per-chunk report (phonemes, unknown words that were split or spelled).
* One level down, `TNNetKokoro.SynthesizeText` (`neural/neuralpretrained.pas`)
  gives you the audio as a `TNNetVolume` instead of a file.

## Running

```
KokoroTTS <model-dir> [options]
```

The examples below use the folders of the Quick start and write
`KokoroTTS` for `neural-api/bin/x86_64-linux/bin/KokoroTTS`.

The model folder is the first argument (as in ChatTerminal). Without `-p` or
`--phonemes` the program opens a REPL: it loads the model once, then each
typed line is spoken into the next WAV.

```
$ KokoroTTS ./k --dict-dir misaki/misaki/data
Model      : ./k
Voice      : af_heart
Dictionary : misaki/misaki/data
Compute    : CPU
Threads    : parallel, every CPU thread (4)
Loaded in  : x.xx s
Type English text; each line is written to the next WAV. /help lists the commands, /exit quits.
> Hello world. This is a test.
Wrote      : kokoro_001.wav (x.xx s of audio, 1 chunk(s))
> /voice bf_emma
Voice      : bf_emma
> Good morning.
Wrote      : kokoro_002.wav (x.xx s of audio, 1 chunk(s))
> /exit
Bye.
```

(x.xx: the times depend on the machine; `--stats` prints them.)

One-shot runs never read stdin and exit with code 0, or 1 on any error
(including a failing `--play-cmd` / `--encode-cmd`):

```
KokoroTTS ./k --dict-dir misaki/misaki/data -p "Hello world. This is a test." --out hello.wav
KokoroTTS ./k --phonemes "həlˈO wˈɜɹld." --out hello.wav
```

### REPL commands

| Command | Effect |
| --- | --- |
| any text | spoken into the next output file |
| `/voice <name>` | switch to `voices/<name>.pt` of the model folder; a change of accent (`a*` <-> `b*`) loads the other dictionaries. On an error (missing file, non-English voice, missing dictionaries) the voice stays. `/voice` alone prints the voice |
| `/speed <x>` | speaking rate, > 0; `/speed` alone prints it |
| `/out <pattern>` | output names from the next line on (the counter continues); `/out` alone prints the pattern and the next name |
| `/phonemes <IPA>` | speak this phoneme string once (no G2P) |
| `/show` | the last line: per chunk the text, the phonemes, the symbols outside the vocab, and the words read by the unknown-word routes (split / spelled) or left unknown |
| `/help` | the command list |
| `/exit` | quit (end of input quits too) |

An unknown command or a failed line prints `Error: ...` and the REPL goes on.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `<model-dir>` | required | snapshot folder (config.json, the single `*.pth`, `voices/`, `misaki/`) |
| `-p TEXT` | REPL | speak one line of text and exit |
| `--phonemes IPA` | off | speak one phoneme string (misaki's G2P alphabet) and exit; no dictionaries needed |
| `--dict-dir DIR` | `<model-dir>/misaki` | misaki's `{us,gb}_{gold,silver}.json` |
| `--voice NAME` | `af_heart` | `voices/NAME.pt` |
| `--speed S` | `1` | speaking rate; durations are divided by S |
| `--out PATTERN` | `kokoro_%03d.wav` | output names, see below |
| `--play-cmd CMD` | none | program run after each WAV, e.g. `"aplay {wav}"` |
| `--encode-cmd CMD` | none | program run after each WAV, e.g. `"lame --quiet {wav} {base}.mp3"` |
| `--keep-wav BOOL` | `true` | `false` deletes the WAV after a successful encode (needs `--encode-cmd`) |
| `--stats` | off | timing to stderr, see below |
| `--serial` | off | single-threaded forward passes |
| `--max-threads N` | every CPU thread | cap the parallel forward at N workers |
| `--gpu` | off (CPU) | stage A on OpenCL (build with `-dOpenCL`, as every build mode of `KokoroTTS.lpi` does); stages B and C and the host steps stay on the CPU |
| `--gpu-platform N`, `--gpu-device N` | 0, 0 | OpenCL platform / device |
| `--no-gpu-shared-kernel` | off | per-layer OpenCL kernels |
| `--selftest` | | offline checks of the argument parsing and the REPL commands (no model), then exit |

`--model-dir` was removed; the folder is the first argument now.

### Output names

`--out` (and `/out`) take a pattern. `%d`, `%Nd` and `%0Nd` (N up to 9) are
replaced by the line counter: 1 for the first WAV written, plus 1 after each
WAV written (a failed line does not count). `%%` is a literal `%`; any other
`%` sequence is rejected. A pattern without a counter is used as is, so every
line overwrites that one file. Existing files are overwritten without a
question.

| Pattern | REPL lines 1, 2, 3 | `-p` / `--phonemes` |
| --- | --- | --- |
| `kokoro_%03d.wav` (default) | `kokoro_001.wav`, `kokoro_002.wav`, `kokoro_003.wav` | `kokoro_001.wav` |
| `take%d.wav` | `take1.wav`, `take2.wav`, `take3.wav` | `take1.wav` |
| `hello.wav` | `hello.wav` each time (overwritten) | `hello.wav` |

### Playing and encoding: external programs

`KokoroTTS` writes only WAV. MP3 (or any other format) needs an external
encoder for now, such as `lame` or `ffmpeg`; a pure Pascal MP3 encoder is not
scheduled yet. Two options run external programs after each WAV is written,
in this order: `--play-cmd`, then `--encode-cmd`, then (with
`--keep-wav false` and only when the encode exited with code 0) the WAV is
deleted. Both are unset by default.

How the command string is run (`SplitCommandTemplate` and
`ExpandCommandArguments` in `KokoroTTS.lpr`; FPC's `TProcess`, no shell):

1. The string is split into arguments at spaces and tabs. `'...'` and
   `"..."` group text, spaces included; the quotes are removed and can sit
   inside an argument (`--title="a b"` gives `--title=a b`). There are no
   escapes; `""` is an empty argument; an unbalanced quote is an error at
   startup.
2. In each argument, `{wav}` becomes the WAV path and `{base}` the WAV path
   without its final `.wav`. A substituted path stays one argument even
   when it contains spaces, and it is not scanned again.
3. The first argument is the program (searched in `PATH`), the rest are its
   arguments. No shell runs: `|`, `>`, `;`, `$VAR`, `*` and `~` are passed
   as literal text. For a pipeline, call a shell yourself:
   `--play-cmd "sh -c 'aplay \"\$0\"' {wav}"`.
4. The program inherits the terminal. `KokoroTTS` waits for it to end. A
   program that cannot start, exits with a non-zero code or is ended by a
   signal prints the reason (`Play command: aplay failed with exit code 1`);
   the REPL goes on, and `-p` exits with code 1.

Examples:

```
# speak each line as soon as it is written (ALSA)
KokoroTTS ./k --dict-dir misaki/misaki/data --play-cmd "aplay -q {wav}"
# MP3 next to each WAV, then drop the WAV
KokoroTTS ./k --dict-dir misaki/misaki/data --encode-cmd "lame --quiet {wav} {base}.mp3" --keep-wav false
# ffmpeg to Opus
KokoroTTS ./k --dict-dir misaki/misaki/data --encode-cmd "ffmpeg -loglevel error -y -i {wav} {base}.opus"
# macOS
KokoroTTS ./k --dict-dir misaki/misaki/data --play-cmd "afplay {wav}"
```

### Timing (`--stats`)

`--stats` prints to stderr, so stdout stays clean: once after loading
(`[stats] load x.xx s (model, voice, dictionaries), peak RSS ... MB`) and per
line: chunks, phonemes, audio seconds, synthesis seconds, the real-time
factor (synthesis time divided by audio time; below 1 is faster than real
time), the WAV write time and the peak RSS.

**Zero noise:** the hn-NSF harmonic source runs without random noise by
default, so the same input gives the same WAV. The holder has a random-noise
option (`HarmonicSource.RandomNoise`, `NoiseSeed`); this example does not
expose it.

Build with `lazbuild -B KokoroTTS.lpi` (modes Default, Debug, Release; all
three define AVX2 and OpenCL). The binary goes to `bin/<cpu>-<os>/bin/`.

## Memory

Measured on the development VM (1 CPU core, real checkpoint, voice
`af_heart`, one process per run under a 2 GB memory cap). KokoroTTS numbers
come from `--stats`; the Python numbers come from the reference `kokoro`
package (`KModel`, FP32, CPU, `torch.set_num_threads(1)`) on the same
phonemes. A frame is 25 ms of audio (40 frames per second).

| Line | Phonemes | Audio | Frames | KokoroTTS peak RSS | Python peak RSS (forward) |
| --- | --- | --- | --- | --- | --- |
| after loading | | | | 462 MB | 1092 MB (1281 MB while loading) |
| 1 x "Hello" | 5 | 1.40 s | 56 | 512 MB | 1110 MB |
| 8 x "Hello" | 47 | 3.20 s | 128 | 586 MB | 1147 MB |
| 26 x "Hello" | 155 | 10.38 s | 415 | 724 MB | 1345 MB |
| the `SimpleKokoroTTS` story | 260 | 15.90 s | 636 | 862 MB | 1354 MB |

* A line adds about 0.6 MB per frame (about 24 MB per second of audio) to
  the 462 MB that loading takes. Python grows unevenly (0.04 to 0.7 MB per
  frame between these lines) on top of a larger base, so KokoroTTS stays
  below Python's peak at every measured length (0.46x to 0.64x).
* Memory follows the longest chunk, not the whole line: text is split into
  chunks of at most 510 phonemes, and `TNNetKokoro.Synthesize` frees stage
  A's net and the other per-chunk buffers before the next chunk starts. A
  full 510-phoneme chunk (about 31 s of audio) has not been measured; from
  the slope above it needs about 1.2 GB (estimate).
* Lines do not add up: in the REPL, a second 26 x "Hello" line raised the
  peak by only 14 MB (724 MB to 738 MB).
* `SimpleKokoroTTS` (`fox.wav`, then `story.wav`) peaked at 872 MB.

Speed on the same VM (1 core): synthesis took 28.8 s for the 10.38 s line
(real-time factor 2.8); Python's forward took 16.8 s.

## Smoke run without the real checkpoint

`tests/fixtures/tiny_kokoro82m` has the same layout with tiny random weights
(the output is noise, which is expected), and `tests/fixtures/g2p_english`
holds small subsets of misaki's dictionaries (enough for the fixture's test
sentences):

```
KokoroTTS tests/fixtures/tiny_kokoro82m --dict-dir tests/fixtures/g2p_english \
  -p "The quick brown fox." --out tiny.wav
printf 'The quick brown fox.\n/show\n/exit\n' | \
  KokoroTTS tests/fixtures/tiny_kokoro82m --dict-dir tests/fixtures/g2p_english
KokoroTTS --selftest
```

## How it works

`KokoroTTS` loads a `hexgrad/Kokoro-82M` snapshot folder (Apache-2.0;
StyleTTS2 + iSTFTNet, 81.7M parameters) into `TNNetKokoro`
(`neural/neuralpretrained.pas`) once:

```
text -> English G2P (neural/neuralg2p.pas, misaki's dictionaries)
  -> chunks of at most 510 phonemes (split at sentence punctuation, as kokoro's KPipeline)
  -> per chunk: phonemes -> ids (config.json vocab, wrapped in 0s)
     -> stage A, three TNNets (PL-BERT/ALBERT, text encoder, duration predictor) -> frames per token
     -> stage B, plain code (F0 and energy curves)
     -> stage-C decoder, plain code
     -> harmonic source + STFT (host)
     -> stage-C generator, plain code -> exp / sin -> inverse STFT (host)
  -> chunk audio joined with no gap -> WAV
```

Stage A is three `TNNet`s run by `TNNet.Compute`: the PL-BERT embeddings,
ALBERT's one shared layer (run 12 times, each output fed back as the next
input) and the rest (text encoder, duration predictor). Their weights are
loaded once, and per chunk `TNNetKokoro` builds the nets for that chunk's
length on those weights (no second copy) and frees them after stage A has
run. Stages B and C
are plain Pascal code on `TNNetVolume` buffers (`neural/neuralplainops.pas`,
called by `TNNetKokoro`): their weights are loaded once, and each step reuses
a few buffers instead of keeping one set per layer. `--gpu` runs stage A on
OpenCL; stages B and C always run on the CPU.

The shared part (loading, voice switching, text -> audio -> WAV) is
`TKokoroSession` in `neural/neuralkokorosession.pas`, so a later TTS server
can reuse it; `KokoroTTS.lpr` holds the argument parsing,
the REPL and the external-program step.

Only the English voices work with text: `a*` (American: `af_*`, `am_*`) and
`b*` (British: `bf_*`, `bm_*`). The other languages need their own G2P; their
voices can still be used with `--phonemes`.

misaki's English dictionaries (Apache-2.0) are `us_gold.json` and
`us_silver.json` (American voices), `gb_gold.json` and `gb_silver.json`
(British voices). The program never downloads them. By default it reads
them from `<model-dir>/misaki/`; `--dict-dir <folder>` points elsewhere. A
missing dictionary file stops the program at startup with the list of the
missing paths. Loading takes about 0.1-0.2 s and 17-21 MB per accent
(measured on the development VM); `/voice` loads the other accent's pair
when the accent changes.

## Parity

Against the reference `kokoro` 0.9.4 package (`tools/make_kokoro82m_fixture.py`):
the tiny fixture per stage in the default test suite (`TestKokoro*Tiny*`), and
the real checkpoint per stage in tests that run only when the snapshot and
the reference dump (`tools/make_kokoro82m_fixture.py real`) are present
(`TestKokoroReal*`; the stage-C decoder, stages-chained, generator and
end-to-end ones also need `NEURAL_KOKORO82M_REAL_HEAVY=1`). End to
end the audio cannot match torch bit for bit: on unvoiced spans some STFT
phase bins of the harmonic source are rounding noise (MKL FFT artefacts in
torch), and the generator amplifies them (expected max abs difference about
0.1).
