# KokoroTTS — Kokoro-82M text-to-speech

Status: **planned (coded), not yet tested by the user on the real
checkpoint.** The code exists and matches the reference `kokoro` package stage
by stage on the real checkpoint (see Parity below), but no full sentence has
been synthesized and listened to on a real machine yet.

`KokoroTTS` loads a `hexgrad/Kokoro-82M` snapshot folder (Apache-2.0;
StyleTTS2 + iSTFTNet, 81.7M parameters) into `TNNetKokoro`
(`neural/neuralpretrained.pas`) once, then speaks English text into 16-bit
mono WAV files at **24 kHz**:

```
text -> English G2P (neural/neuralg2p.pas, misaki's dictionaries)
  -> chunks of at most 510 phonemes (split at sentence punctuation, as kokoro's KPipeline)
  -> per chunk: phonemes -> ids (config.json vocab, wrapped in 0s)
     -> stage A  (PL-BERT/ALBERT, text encoder, duration predictor) -> frames per token
     -> stage B  (F0 and energy curves)
     -> stage-C decoder
     -> harmonic source + STFT (host)
     -> stage-C generator -> exp / sin -> inverse STFT (host)
  -> chunk audio joined with no gap -> WAV
```

Stages A, B and C are `TNNet`s run by `TNNet.Compute`; each is built once
from the checkpoint and resized per chunk by a borrowing rebuild (no second
copy of the weights). The shared part (loading, voice switching, text ->
audio -> WAV) is `TKokoroSession` in `neural/neuralkokorosession.pas`, so a
later TTS server can reuse it; `KokoroTTS.lpr` holds the argument parsing,
the REPL and the external-program step.

Only the English voices work with text: `a*` (American: `af_*`, `am_*`) and
`b*` (British: `bf_*`, `bm_*`). The other languages need their own G2P; their
voices can still be used with `--phonemes`.

## Download

From `https://huggingface.co/hexgrad/Kokoro-82M` you need:

- `config.json`
- `kokoro-v1_0.pth` (327 MB)
- `voices/<name>.pt` for each voice you want (e.g. `voices/af_heart.pt`,
  0.5 MB)

```
huggingface-cli download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth \
  voices/af_heart.pt voices/bf_emma.pt --local-dir Kokoro-82M
```

For text you also need misaki's English dictionaries (Apache-2.0) from
`https://github.com/hexgrad/misaki`, folder `misaki/data/`:
`us_gold.json` and `us_silver.json` (American voices), `gb_gold.json` and
`gb_silver.json` (British voices). The program never downloads them. By
default it reads them from `<model-dir>/misaki/`; `--dict-dir <folder>`
points elsewhere.

```
mkdir -p Kokoro-82M/misaki
for f in us_gold us_silver gb_gold gb_silver; do
  wget -O Kokoro-82M/misaki/$f.json \
    https://raw.githubusercontent.com/hexgrad/misaki/main/misaki/data/$f.json
done
```

The same files ship inside the `misaki` Python package
(`site-packages/misaki/data/`), if it is installed somewhere.

A missing dictionary file stops the program at startup with the list of the
missing paths. Loading takes about 0.1-0.2 s and 17-21 MB per accent
(measured on the development VM); `/voice` loads the other accent's pair when
the accent changes.

## Running

```
KokoroTTS <model-dir> [options]
```

The model folder is the first argument (as in ChatTerminal). Without `-p` or
`--phonemes` the program opens a REPL: it loads the model once, then each
typed line is spoken into the next WAV.

```
$ KokoroTTS Kokoro-82M
Model      : Kokoro-82M
Voice      : af_heart
Dictionary : Kokoro-82M/misaki
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

(x.xx: the times depend on the machine; none was measured on the real
checkpoint yet.)

One-shot runs never read stdin and exit with code 0, or 1 on any error
(including a failing `--play-cmd` / `--encode-cmd`):

```
KokoroTTS Kokoro-82M -p "Hello world. This is a test." --out hello.wav
KokoroTTS Kokoro-82M --phonemes "həlˈO wˈɜɹld." --out hello.wav
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
| `--gpu` | off (CPU) | stage nets on OpenCL (build with `-dOpenCL`, as every build mode of `KokoroTTS.lpi` does); host steps stay on the CPU |
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
KokoroTTS Kokoro-82M --play-cmd "aplay -q {wav}"
# MP3 next to each WAV, then drop the WAV
KokoroTTS Kokoro-82M --encode-cmd "lame --quiet {wav} {base}.mp3" --keep-wav false
# ffmpeg to Opus
KokoroTTS Kokoro-82M --encode-cmd "ffmpeg -loglevel error -y -i {wav} {base}.opus"
# macOS
KokoroTTS Kokoro-82M --play-cmd "afplay {wav}"
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

No full sentence has been run on the real checkpoint on a measured machine
yet. What was measured on the development VM (CPU, real checkpoint):

- stages A -> B -> stage-C decoder on a 102-frame sentence (2.55 s of audio):
  999 MB peak RSS for the test process;
- the stage-C generator is the largest activation user: about 4.1 MB per
  frame on a stage-C-like test net, so a long input can need several GB. An
  earlier real generator run with host-output sharing off was killed at
  2.7 GB RSS on the 3.9 GB VM.

Text is split into chunks of at most 510 phonemes, and each chunk builds its
own stage nets, so the activation peak follows the longest chunk, not the
whole line. A full 510-phoneme chunk has not been measured; keep lines to a
sentence or two until the end-to-end peak is measured.

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
