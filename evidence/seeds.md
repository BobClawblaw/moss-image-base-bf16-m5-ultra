# Whole pictures against a float32 run of the schedule

Each case runs the card's schedule three ways from the same prompt rows and noise: the publisher's pipeline in torch
bf16 (MPS), the same pipeline with the transformer in float32 (`moss_image_case.py exact`), and ours (engine
0b69e127). Each picture is compared with the float32 one (PSNR, dB; higher is closer).

| prompt | size | seed | torch bf16 | ours | closer |
|---|---|---|---|---|---|
| fox | 1024x1024 | 42 | 40.79 | 41.16 | ours |
| sticky note | 1200x880 | 7 | 32.08 | 27.41 | torch |
| fox | 1024x1024 | 1 | 38.66 | 41.37 | ours |
| fox | 1024x1024 | 2 | 26.68 | 26.69 | ours |
| fox | 1024x1024 | 3 | 24.81 | 24.66 | torch |
| sticky note | 1200x880 | 1 | 28.52 | 37.05 | ours |
| sticky note | 1200x880 | 2 | 28.66 | 28.79 | ours |
| sticky note | 1200x880 | 3 | 31.13 | 31.28 | ours |
| mean | | | 31.42 | 32.30 | ours in 6 of 8 |

Both bf16 runs land about as far from exact arithmetic on average; which is closer changes from case to case,
as rounding of equal size but different placement would. `fridge-zh-fp32-steps/run.json` holds the per-step view:
both branches of all 50 steps of the 1200x880 seed-7 case against float32 velocities on the same inputs.
