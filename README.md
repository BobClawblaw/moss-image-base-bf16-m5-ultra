# MOSS-Image-Base on one M5 Ultra (TensorFold Metal, bf16 as published)

Makes pictures with [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), OpenMOSS's 5B
flow-matching text-to-image model, from its release as published. The 5B joint transformer, which does nearly all
the work, and the prompt encoder (the checkpoint's Qwen3.5 text model) run on TensorFold's Metal runtime (the
`moss_image` family). The image decoder (FLUX.2's small decoder) and the starting noise run in torch, through the
publisher's own `pipeline_moss_image.py`. Nothing is converted: the transformer and encoder are bf16 and the decoder
float32, as released.

The torch steps import and run Python from the checkpoint itself: its `pipeline_moss_image.py`, through
`tools/zig/moss_image_case.py` in the engine's tree. That tool checks the file's sha256 against revision 85975793's
(`1dac1aca…8b31`) before importing it, and refuses any other file. Download the checkpoint at that revision (below).

The transformer is on [TensorFold](https://github.com/ashhart/TensorFold) main (#1457); the encoder is on branch
`moss-image-encoder` of [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold), not yet on main. This
recipe builds its pinned commit. It runs one picture at a time from the command line; there is no server.

![The fox prompt at 1024x1024, seed 42: the publisher's torch pipeline (left), ours (right)](evidence/fox-1024x1024-42/side.png)

More pictures, with their prompts and seeds, are in the [gallery](evidence/gallery/README.md).

## Quick start

Needs an M5 Mac: the engine runs on the M5's tensor units and refuses an earlier chip, which was never checked. Also
Python 3.11+, git, `hf` and [Zig 0.17.0](https://ziglang.org/download/), and about 26 GB of disk for the checkpoint.

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download OpenMOSS-Team/MOSS-Image-Base --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 --local-dir ~/models/MOSS-Image-Base
cp local.example.toml local.toml      # then set zig, python (3.11+) and paths, if yours differ
python3 -m mac setup                  # torch 2.14.1, diffusers 0.42.0, transformers 5.21.0 in a venv under ~/.cache/moss-image
python3 -m mac build                  # the pinned engine's tf-moss-image
python3 -m mac generate "A red fox sitting in fresh snow at dawn" -o fox.png
```

`python3 -m mac status` shows the pins and what is built; `python3 -m mac check PROMPT_FILE` compares the
publisher's pipeline with ours (below).

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`STEPS=30 python3 -m mac generate "..."`). `generate` also takes them as flags.

| setting | default | |
|---|---|---|
| `width`, `height` | 1024, 1024 | positive multiples of 16 |
| `steps` | 50 | Euler steps on the scheduler's shifted grid (shift 5) |
| `seed` | 42 | the starting noise, drawn in float32 on the CPU as the pipeline draws it |
| `negative` | empty | the negative prompt |
| `device` | mps | where torch runs the decoder |

The guidance is the card's and fixed: scale 3, two transformer passes a step, the negative one skipping block 22.

**Time and memory** for a 1024x1024 picture, three prompts on the pinned engine with no other GPU work: 1.9 to 2.3 s
for torch to draw the starting noise and the schedule, then the engine's process (49 to 50 s): 0.6 s to load the
encoder and 0.24 to 0.26 s to encode both prompts, then 47 to 48 s for the transformer's 50 steps (100 passes). Then
2.4 to 2.5 s for torch to decode. The engine's process peaks at a 16.7 GB footprint, the encoder's 16 GB of bf16
weights; it frees them before loading the transformer (9.5 GB, 11.3 GB at its peak).

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), checkpoint revision 85975793. The transformer was measured on engine commit
772283b6; TensorFold main's 2551c573, then 6f05bcd9 (which adds the encoder), 198ed906 and the pinned 92c82988, give
the identical latents from torch's prompt rows. The reference is the checkpoint's own pipeline on torch 2.14.1 with
MPS in bf16, run by `tools/zig/moss_image_case.py` in the engine's tree. MPS has no float64, so the case tool computes
the time sinusoid (float64 by the publisher) on the host; every other step runs unchanged.

Two cases at the card's defaults:
- A fox at 1024x1024, seed 42 (`cases/fox.txt`).
- The card's Chinese sticky-note prompt at 1200x880, seed 7 (`cases/fridge-zh.txt`): an odd 55 x 75 latent grid and
  63 prompt rows.

- **The prompt encoder.** Its token ids equal the publisher's tokenizer's for the empty negative prompt (10), the fox
  (34) and the sticky note (63). Every layer's rows reach cosine 0.99974 or better against torch bf16's. Its final
  rows against a float32 run of the encoder: cosine 0.999826, 0.999931 and 0.999926 (torch bf16's own: 0.999806,
  0.999934, 0.999938). It encodes a prompt in 0.054 to 0.068 s, against torch MPS's 0.157 to 0.163 s, and loads in
  0.6 s against 2.3 s.
- **Each step.** Both branches of each of the 50 steps run on the pipeline's own input and are compared with its
  velocities. Cosine: 0.99973 to 0.99998 on torch's prompt rows, 0.99972 or better on ours.
- **Against exact arithmetic.** A float32 run of the same transformer on the same inputs, at every step of the
  sticky-note case. Our distance from it (mean 1 - cosine 1.602e-4 prompt, 1.735e-4 negative) is the same size as
  torch bf16's own (1.602e-4, 1.715e-4), and ours is the closer at 25 and 27 of the 50 steps.
- **Pictures.** Ours against the pipeline's: 47.07 dB PSNR for the fox, 28.71 dB for the sticky note (48.52 and
  28.93 dB with torch's prompt rows). Fifty guided
  steps magnify bf16 rounding into different fine detail (the note's text and the scene stay; a cabinet edge moves),
  so the ruler is a float32 run of the whole schedule. Over 8 cases (both prompts, seeds 42/7, 1, 2, 3), with the
  transformer in float32 on torch's bf16 encoder rows, ours is the closer to it in 6, at a mean of 32.3 dB PSNR against
  torch bf16's 31.4 dB. With the encoder in float32 too, torch bf16 averages 32.5 dB and ours 31.2 dB: six cases are
  within 0.5 dB or favor ours, and two sticky-note seeds favor torch's by 4.6 and 8.7 dB. On that same ruler, ours
  with its own encoder averages 31.1 dB (closer than torch's rows with our transformer in 4 of the 8). That ruler is
  `moss_image_case.py exact --fp32-encoder`. The first ruler's rows are in [evidence/seeds.md](evidence/seeds.md).
- **Repeats.** The same step run twice with another between gives identical bytes; `generate` and `check` give the
  identical PNG for the same prompt and seed.
- **Speed**, one session with no other GPU work: 0.489 s a transformer pass at 1024x1024 against torch MPS's 0.744 s
  (1.52x); 0.486 s against 0.760 s at 1200x880 (1.56x). Ours is the median of three runs of 50 guided steps, torch's
  of its 100 passes.
- **Kernels.** Every kernel of the family passes TensorFold's kernel contract (gates 1-4 on this GPU, gate 5's
  limits), and sixteen planted controls go red: `SUMMARY 22 kernels: 22 pass gates 1-4, 0 fail, 0 known findings;
  16 controls: 16 red as planted, 0 missed; 0 limit problems`.

`evidence/` holds each case's `summary.json`, the engine's `run.json` (every step's comparison and timing), the
reference's `reference.json` and the side-by-side pictures.

## What it does not do

- **The decoder is not ported here.** It runs in torch (MPS), 2.5 s a picture; a native decoder is a separate branch.
- **No server.** Pictures come from the command line, one at a time.
- **Quality suites:** no image-quality benchmark was run (GenEval, DPG and the like). The checks above show the
  transformer computes what the publisher's pipeline computes, to bf16 rounding; the model's quality is the
  publisher's.
- **Batches, LoRA and the extra skip-layer guidance** (`slg_layers`) are not wired. The SFT checkpoint is untested.

## Pins

| | |
|---|---|
| checkpoint | `OpenMOSS-Team/MOSS-Image-Base` @ `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` (Apache 2.0) |
| engine | BobClawblaw/TensorFold @ `92c829887ed66c685ab800e36376a18bc1ddf44d` (branch `moss-image-encoder`) |
| torch side | torch 2.14.1, diffusers 0.42.0, transformers 5.21.0 |
| Zig | 0.17.0 |
