# MOSS-Image-Base on one M5 Ultra (TensorFold Metal, bf16 as published)

Makes pictures with [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), OpenMOSS's 5B
flow-matching text-to-image model, from its release as published. The 5B joint transformer, which does nearly all
the work, runs on TensorFold's Metal runtime (the `moss_image` family). The prompt encoder (a Qwen3.5 model) and the
image decoder (FLUX.2's small decoder) run in torch, through the publisher's own `pipeline_moss_image.py`. Nothing is
converted: the transformer and encoder are bf16 and the decoder float32, as released.

The family is on branch `moss-image` of [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold); it is
not on TensorFold main. This recipe builds its pinned commit. It runs one picture at a time from the command line;
there is no server.

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
| `device` | mps | where torch runs the encoder and the decoder |

The guidance is the card's and fixed: scale 3, two transformer passes a step, the negative one skipping block 22.

**Time and memory** for a 1024x1024 picture: 5 s to load and run the prompt encoder (7 s the first time), 48 s for
the transformer's process (47 s for its 50 steps, 100 passes), 2.4 s to decode. The transformer process peaks at an
11.3 GB footprint (its bf16 weights are 9.5 GB). The encoder runs before it in its own torch process, which loads the encoder's 16 GB of bf16 weights; its
peak was not measured.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), engine commit 772283b6, checkpoint revision 85975793. The reference is the
checkpoint's own pipeline on torch 2.14.1 with MPS in bf16, run by `tools/zig/moss_image_case.py` in the engine's
tree. MPS has no float64, so the case tool computes the time sinusoid (float64 by the publisher) on the host; every
other step runs unchanged.

Two cases at the card's defaults:
- A fox at 1024x1024, seed 42 (`cases/fox.txt`).
- The card's Chinese sticky-note prompt at 1200x880, seed 7 (`cases/fridge-zh.txt`): an odd 55 x 75 latent grid and
  63 prompt rows.

- **Each step.** Both branches of each of the 50 steps run on the pipeline's own input and are compared with its
  velocities. Cosine: 0.99973 to 0.99998.
- **Against exact arithmetic.** A float32 run of the same transformer on the same inputs, at every step of the
  sticky-note case. Our distance from it (mean 1 - cosine 1.602e-4 prompt, 1.735e-4 negative) is the same size as
  torch bf16's own (1.602e-4, 1.715e-4), and ours is the closer at 25 and 27 of the 50 steps.
- **Pictures.** Ours against the pipeline's: 48.52 dB PSNR for the fox, 28.93 dB for the sticky note. Fifty guided
  steps magnify bf16 rounding into different fine detail (the note's text and the scene stay; a cabinet edge moves),
  so the ruler is a float32 run of the whole schedule. Over 8 cases (both prompts, seeds 42/7, 1, 2, 3), ours is the
  closer to it in 6, at a mean of 32.3 dB PSNR against torch bf16's 31.4 dB. Every row is in
  [evidence/seeds.md](evidence/seeds.md). Those 8 ran on the earlier build 0b69e127; 772283b6 gives byte-identical
  latents on the fox and the sticky note, and the identical gallery pictures.
- **Repeats.** The same step run twice with another between gives identical bytes; `generate` and `check` give the
  identical PNG for the same prompt and seed.
- **Speed**, one session with no other GPU work: 0.489 s a transformer pass at 1024x1024 against torch MPS's 0.744 s
  (1.52x); 0.486 s against 0.760 s at 1200x880 (1.56x). Ours is the median of three runs of 50 guided steps, torch's
  of its 100 passes.
- **Kernels.** Every kernel of the family passes TensorFold's kernel contract (gates 1-4 on this GPU, gate 5's
  limits), and eight planted controls go red: `SUMMARY 11 kernels: 11 pass gates 1-4, 0 fail, 0 known findings;
  8 controls: 8 red as planted, 0 missed; 0 limit problems`.

`evidence/` holds each case's `summary.json`, the engine's `run.json` (every step's comparison and timing), the
reference's `reference.json` and the side-by-side pictures.

## What it does not do

- **The prompt encoder and the decoder are not ported.** They run in torch (MPS). The transformer takes about 86% of
  a picture's time (48 of 56 s); porting the encoder (TensorFold has a `qwen3_5` family) and the decoder is next.
- **No server.** Pictures come from the command line, one at a time.
- **Quality suites:** no image-quality benchmark was run (GenEval, DPG and the like). The checks above show the
  transformer computes what the publisher's pipeline computes, to bf16 rounding; the model's quality is the
  publisher's.
- **Batches, LoRA and the extra skip-layer guidance** (`slg_layers`) are not wired. The SFT checkpoint is untested.

## Pins

| | |
|---|---|
| checkpoint | `OpenMOSS-Team/MOSS-Image-Base` @ `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` (Apache 2.0) |
| engine | BobClawblaw/TensorFold @ `772283b6b38d2f274e33e733df089ef2d6edc882` (branch `moss-image`) |
| torch side | torch 2.14.1, diffusers 0.42.0, transformers 5.21.0 |
| Zig | 0.17.0 |
