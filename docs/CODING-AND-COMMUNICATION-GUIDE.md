# Coding and Communication Guide

A running list of rules for how agents work in this repository and how they
report that work back. `docs/OPTIMIZATION-GUIDE.md` says what fast code looks
like; this guide says when to write code at all, what to name it, where to put
it, how to run experiments, and how to describe the result.

Each entry states the problem, the rule, and why it matters. Rules are added
incrementally as we discover them.

## Menu

| # | Rule | Scope |
| --- | --- | --- |
| [1](#1-do-not-write-code-unless-asked-or-authorized) | Do not write code unless asked or authorized | process |
| [2](#2-names-must-be-meaningful-and-precise--variables-types-and-routines) | Names must be meaningful and precise | code |
| [3](#3-before-adding-code-look-for-code-that-already-does-it) | Before adding code, look for code that already does it | code |
| [4](#4-place-a-new-method-at-the-right-level--parent-class-first) | Place a new method at the right level — parent class first | code |
| [5](#5-reuse-instead-of-copypaste) | Reuse instead of copy/paste | code |
| [6](#6-comment-budget--at-most-two-lines-per-routine-declaration) | Comment budget — at most two lines per routine declaration | code |
| [7](#7-run-every-experiment-under-ulimit-with-a-3-gb-cap) | Run every experiment under `ulimit` with a 3 GB cap | experiments |
| [8](#8-poll-a-running-experiment-every-minute--never-block-on-a-timeout) | Poll a running experiment every minute — never block on a timeout | experiments |
| [9](#9-write-to-the-user-in-clear-technical-language) | Write to the user in clear technical language | communication |
| [10](#10-name-the-actor) | Name the actor — no actorless passive | communication |
| [11](#11-repeat-the-defined-term-never-vary-it-for-style) | Repeat the defined term; never vary it for style | communication |
| [12](#12-watch-for-colliding-terms) | Watch for colliding terms | communication |
| [13](#13-prefer-the-ordinary-word) | Prefer the ordinary word | communication |

## 1. Do not write code unless asked or authorized

**Problem.** An agent asked to *investigate*, *review*, *explain*, or *plan*
answers by editing files. The user now has a diff they did not request, mixed in
with the answer they did request, and has to review both.

**Rule.** Editing the working tree is a separate permission from thinking about
it. Write code only when the request asks for it ("fix", "add", "implement",
"optimize this"), or when the user has authorized the change in this
conversation. When an investigation turns up something worth changing, say what
you would change and why, then wait.

Two cases that are *not* authorization:

- The user approved a change to file A; that does not extend to file B.
- The fix looks obvious and small. Obvious-and-small is exactly the class of
  change a user wants to see proposed first, because it is cheap to describe.

**Why.** The user decides what enters the codebase. An unrequested edit costs
them a review they did not budget for, and it hides the answer they actually
wanted inside a diff.

## 2. Names must be meaningful and precise — variables, types, and routines

**Problem.** `n`, `tmp`, `Value2`, `DoIt`, `TObj2` force every later reader to
re-derive what the code holds. Worse are names that are meaningful but *wrong*
after an edit: a `MaxNeuronPos` that now bounds channels.

**Rule.** A name states what the thing *is*, in the vocabulary of this codebase:

- **Variables** name the quantity, not the mechanism: `TokenCount`,
  `MaxTokenPos`, `HeadStartOffset` — not `Cnt`, `Lim`, `Ofs`.
- **Loop bounds** follow the existing convention: `Max<Entity>Pos` for an
  inclusive `for` bound, `Max<Entity><X|Y|D>` for an axis bound. Never
  invented one-off names.
- **Types** carry the project prefix and the concept: `TNNetFusedSDPA`, not
  `TFastAttn`.
- **Routines** name the effect, and the name must match the effect exactly.
  `ForceOutputOnRAM` moves data; a routine called `Get…` must not mutate.
- **Booleans** read as a predicate: `HasSharedKernel`, `IsQuantized`.

Precision includes the type. Pick the type that states the intent —
`TNeuralFloat` for tensor math, `integer` for counts and offsets — and do not
widen hot math to `Double` (Optimization Guide #25).

**Why.** Names are the only documentation that cannot go stale silently: a wrong
name misleads on every read. In a codebase this size, the name is how the next
agent finds the code at all.

## 3. Before adding code, look for code that already does it

**Problem.** A second BPE encoder, a third bilinear-resize helper, a private
`SoftMax` inside one layer. Each new copy is a place where a future fix will be
applied to the other one.

**Rule.** Before writing a new routine, class, or helper, search for it. Grep
the concept, the likely name, and the likely call site. `TNNetVolume` in
`neural/neuralvolume.pas` already has most elementwise and reduction math; the
loaders, tokenizers, and samplers already have most of their supporting
primitives.

If something close exists but does not quite fit, prefer extending it (an extra
parameter, an overload, a generalized bound) over cloning it. If you conclude
nothing exists, say so in the commit message — that sentence is the record that
the search happened.

**Why.** Duplicated logic diverges. Every fix, every optimization, and every
bug then has to be found twice.

## 4. Place a new method at the right level — parent class first

**Problem.** A method lands on the concrete layer that needed it, and then the
next three layers need it too.

**Rule.** Before adding a method to a class, ask where it belongs:

- Does it use only members the **parent** class has? Then it belongs on the
  parent, where every sibling gets it.
- Is it about volume data rather than about the layer? Then it is a
  `TNNetVolume` method (and it needs both an AVX path and a plain-Pascal
  fallback — see the Optimization Guide's authoring section).
- Is it used only inside one method, and only by it? Then it is a nested
  routine, not a new public surface.

Push the method as high as it is *correct* to push, and no higher: a method on
a parent that only one child can legally call is a worse home than a method on
the child.

**Why.** The right level makes the next layer that needs the behaviour a zero-
line change. The wrong level either duplicates the method across siblings or
puts a trap in the parent's public API.

## 5. Reuse instead of copy/paste

**Problem.** Two loops that differ only in which axis they walk, or two loaders
that differ only in the dtype they decode. They start identical and drift.

**Rule.** When you catch yourself pasting a block and editing two tokens,
factor it instead: a parameter, a shared helper, or a bulk `TNNetVolume` call.
Prefer one general routine over two near-copies.

The exception is the case the Optimization Guide already names: kernel paths
that look like duplication (`AVX32` and plain `AVX`, or an AVX path beside its
Pascal fallback) are deliberate and must stay separate.

**Why.** Near-copies are the main source of "we fixed that already" bugs in
this repository — the fix went to one copy.

## 6. Comment budget — at most two lines per routine declaration

**Problem.** A 100K-line Pascal unit with three comment lines per code line is
a 300K-line unit. Nobody reads it, and the comments rot faster than the code.

**Rule.**

- At most **two lines of comment** on a routine declaration, stating what it
  does and any non-obvious contract (units, ownership, "caller must have called
  `SetPrevLayer`").
- Inside a routine, comment only what the code cannot say: a numerical
  subtlety, a reference to a paper's equation, a deliberate deviation from the
  obvious implementation.
- Do **not** comment what the line already states. `// increment I` is noise.
- Do **not** write history in comments. "was 1024, changed to 512" belongs in
  the commit message; a comment states the *current* rationale.

If a routine needs a long explanation to be understandable, that is usually a
naming problem (#2) or a decomposition problem, not a comment problem.

**Why.** Comment volume is a maintenance cost paid on every read and every
edit, and stale comments are worse than no comments.

## 7. Run every experiment under `ulimit` with a 3 GB cap

**Problem.** A benchmark or a model load that allocates without bound takes the
whole box down. On this machine that costs the user their session, not just the
experiment.

**Rule.** Wrap any experiment — benchmark, example binary, model load, test
sweep — in a 3 GB address-space cap:

```bash
( ulimit -v 3145728; ./MyExperiment )
```

The subshell keeps the limit local to the experiment. A run that dies at the
cap is a **result**: report it as "exceeded 3 GB" rather than retrying without
the limit. Raising or removing the cap needs the user's agreement first.

**Why.** A capped failure is information about memory use. An uncapped failure
is an unresponsive machine.

## 8. Poll a running experiment every minute — never block on a timeout

**Problem.** An agent starts a 30-minute run with a 30-minute timeout and then
does nothing for 30 minutes. If the run wedged in the first 20 seconds, the user
waited 30 minutes to learn that.

**Rule.** Start long work in the background, then check it about **once a
minute**: is the process alive, is the output file still growing, has it printed
an error? Report the first real signal as soon as you see it — a crash, a stall,
a first token, a first epoch — instead of waiting for the exit code.

Stop early when the answer is already in: if the experiment exists to compare A
and B and A has clearly lost by minute two, say so and stop the run.

**Why.** User attention is the scarce resource here, not CPU time. A minute-
granularity report turns a 30-minute wait into a 30-second answer whenever
something goes wrong, which is most of the time.

## 9. Write to the user in clear technical language

**Problem.** Two opposite failures. One is vague ("improved performance a bit,
should be better now"). The other is unreadable — a wall of jargon, nested
clauses, and numbers with no units.

**Rule.** Be technical *and* plain:

- **Name the thing.** `TNNetFusedSDPA.Compute`, `neuralvolume.pas:1420`, not
  "the attention code".
- **Quantify with units.** "2.2 ms/token, down from 13.9 ms/token" beats
  "much faster". Say what was measured and on what.
- **One idea per sentence.** Short sentences, ordinary words, technical terms
  where they are the precise term — not to sound thorough.
- **Lead with the result**, then the detail. The user should get the answer in
  the first sentence and the evidence after it.
- **State what you did not do.** Skipped tests, unverified assumptions, and
  parts of the request you left out are part of the report, not omissions.
- **Do not hedge a verified result**, and do not present an unverified one as
  fact. "Tests pass (2729/2729)" or "not run" — never "should work".

**Why.** The user is reading to make a decision. Vague prose forces another
round-trip; dense prose forces them to parse instead of decide.

## 10. Name the actor

**Problem.** A passive verb with no actor is the easiest way to lose a fact
without noticing. "The weights are quantized", "the output is moved to RAM",
"the cache is allocated" — each drops the one thing the reader needs in order
to know where to look. The sentence still sounds complete, so nobody asks.

**Rule.** Every claim about behaviour names who does it. Say which routine,
which layer, which caller:

| Actorless | Named actor |
| --- | --- |
| "The weights are quantized." | "`TNNet.BuildQuantInt8` quantizes the weights." |
| "The output is moved to RAM." | "The caller must call `ForceOutputOnRAM` before reading `Output`." |
| "The KV cache is allocated." | "`Begin` allocates the KV cache, so the int8 flag must be set before it runs." |
| "The value is validated." | "The loader rejects non-finite rows; the quantizer tolerates them." |

The same applies to your own work: "the test was fixed" hides whether you fixed
the test or the code under it.

**Why.** The actor is the address. Without it the reader cannot find the code,
cannot tell whose responsibility a step is, and cannot tell whether it happens
automatically or only if someone calls it — which is exactly the bug class this
codebase keeps producing (a flag set after the allocation that reads it).

## 11. Repeat the defined term; never vary it for style

**Problem.** Prose style teaches you not to repeat a word. Technical writing
wants the opposite. A paragraph that says "volume", then "tensor", then "the
buffer" makes the reader stop and work out whether that is one object or three.

**Rule.** Once a term is defined, it appears in that exact form every time it is
meant, however repetitive that reads. `TNNetVolume` stays `TNNetVolume` — not
"the tensor", not "the array", not "the data". A **layer** is a layer; a
**neuron** is a neuron; the **chunk path** is the chunk path.

Corollary: if you do use a different word, the reader is entitled to assume you
mean a different thing. So introducing a synonym is a bug in the text, and
introducing a genuinely new concept requires a new name, not a reused one.

**Why.** Consistency is what makes a term searchable and checkable. A reader
who cannot tell whether two words name one thing or two has to reconstruct the
model you already had.

## 12. Watch for colliding terms

**Problem.** The same word means different things in two nearby contexts, and
the document never says so. This repository is full of live collisions:

- **model** — the `TNNet` in memory, the checkpoint on disk (`Qwen2.5-7B`), and
  the LLM running the agent.
- **kernel** — an OpenCL kernel, a convolution kernel, and a SIMD "kernel path"
  (`AVX32` versus plain `AVX`).
- **layer** — a `TNNetLayer` object, and a transformer block made of several of
  them.
- **head** — an attention head, and the LM head.
- **weights** — the `TNNetVolume` of parameters, and the int8-quantized copy.

**Rule.** When two meanings of one word are live in the same document or the
same reply, name the distinction once, explicitly, at first use — then keep
each sense in its own consistent form (rule 11). "Model" alone is ambiguous
here; "checkpoint file" and "the loaded `TNNet`" are not.

The same holds in code: two fields that both want the name `FWeights` are the
signal that one of them needs a qualifier (`FWeightsInt8`), not a comment.

**Why.** A collision left unnamed does not read as ambiguous — it reads as
clear and means the wrong thing. That is worse than obvious vagueness, because
the reader has no cue to slow down.

## 13. Prefer the ordinary word

**Problem.** Borrowed terminology asks the reader to work through a metaphor
before reaching the fact. "Contract" is real terminology — design by contract
comes from Eiffel — but it makes the reader picture two parties signing
something in order to learn that renaming a field compiles cleanly and breaks
every remote client at run time.

**Rule.** Name the thing, then state in plain words the part the metaphor was
carrying:

| Metaphor | Name the thing | The fact it was carrying |
| --- | --- | --- |
| "the API contract" | the JSON field names in the body; the wire format when the URL path is included | renaming a field compiles cleanly and breaks every remote client at run time |
| "arm the int8 KV cache" | set `pKVInt8` before `Begin` | `Begin` allocates the cache; setting the flag afterwards does nothing |
| "first-class layer" | a `TNNetLayer` descendant registered for load/save | otherwise the network reloads without it |
| "source of truth" | the `.lpi` file | `lazbuild` reads its defines; a hand-rolled `fpc` line compiles a different program |

Established terms of the field stay: convolution, softmax, quantization, hoist,
strength reduction. The test is whether the word carries the meaning or replaces
it — a term the reader must already share is fine; a metaphor they must unpack
is not.

**Why.** The metaphor is always shorter than the fact and always less useful.
A reader who does not share it gets nothing; a reader who does share it still
has to translate back to what the code actually does.
