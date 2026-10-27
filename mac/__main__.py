"""MOSS-Image-Base on an M5 Mac: the transformer on TensorFold's Metal runtime, the encoder and decoder in torch.

  python3 -m mac setup                     torch, diffusers and transformers (pinned) in a venv under tf_cache
  python3 -m mac build                     the pinned engine's tf-moss-image, under tf_cache
  python3 -m mac generate "PROMPT" [-o out.png] [--width W --height H --steps N --seed S --negative TEXT]
  python3 -m mac check PROMPT_FILE [--exact]   the publisher's pipeline against ours, step by step and as pictures
  python3 -m mac status                    pins, paths and what is built
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import ROOT, Settings, load

# The versions every number in the README was measured with.
PACKAGES = ["torch==2.14.1", "diffusers==0.42.0", "transformers==5.21.0", "accelerate>=1.10", "safetensors>=0.6",
            "pillow>=10", "numpy"]


def run(cmd: list, **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def tool(s: Settings, *args) -> list:
    """The case tool under the venv's python, isolated from this folder's modules."""
    return [s.venv_python, "-I", s.case_tool, *args]


def need(s: Settings) -> None:
    missing = [what for what, p in (("the model (hf download, see README)", s.model / "transformer" / "config.json"),
                                    ("the venv (python3 -m mac setup)", s.venv_python),
                                    ("the engine (python3 -m mac build)", s.engine),
                                    ("the case tool (python3 -m mac build)", s.case_tool)) if not p.exists()]
    if missing:
        sys.exit("missing: " + "; ".join(missing))


def setup(s: Settings, _args) -> int:
    venv = s.cache / "venv"
    if not s.venv_python.exists():
        run([s.python, "-m", "venv", venv])
    run([s.venv_python, "-m", "pip", "install", "-q", *PACKAGES])
    return 0


def build(s: Settings, _args) -> int:
    src = s.path(s.tensorfold_src)
    if not s.worktree.exists():
        s.worktree.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", src, "worktree", "add", "--detach", s.worktree, s.tensorfold_sha])
    run([s.path(s.zig) if "/" in s.zig else s.zig, "build", "tf-moss-image", f"-Dcpu={s.zig_cpu}"], cwd=s.worktree)
    print(s.engine)
    return 0


def generate(s: Settings, args) -> int:
    need(s)
    out = Path(args.out).resolve()
    size = ["--width", str(args.width), "--height", str(args.height)]
    with tempfile.TemporaryDirectory(prefix="moss-image-") as tmp:
        prompt = Path(tmp) / "prompt.txt"
        prompt.write_text(args.prompt)
        mark = time.perf_counter()
        run(tool(s, "start", s.model, tmp, "--prompt-file", prompt, "--negative", args.negative, "--seed",
                 str(args.seed), "--steps", str(args.steps), "--device", s.device, *size))
        encoded = time.perf_counter()
        run([s.engine, s.model / "transformer", Path(tmp) / "case.safetensors", tmp])
        denoised = time.perf_counter()
        run(tool(s, "decode", s.model, Path(tmp) / "latents.f32", out, "--device", s.device, *size))
        done = time.perf_counter()
        timing = json.loads((Path(tmp) / "run.json").read_text())
    print(json.dumps({"image": str(out), "encode_s": round(encoded - mark, 1), "transformer_s": timing["denoise_s"],
                      "denoise_process_s": round(denoised - encoded, 1), "decode_s": round(done - denoised, 1)}))
    return 0


def check(s: Settings, args) -> int:
    need(s)
    name = Path(args.prompt_file).stem
    out = (ROOT / "out" / f"check-{name}-{args.width}x{args.height}-{args.seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    size = ["--width", str(args.width), "--height", str(args.height)]
    run(tool(s, "case", s.model, out, "--prompt-file", Path(args.prompt_file).resolve(), "--seed", str(args.seed),
             "--steps", str(args.steps), "--device", s.device, *size))
    with open(out / "engine.log", "w") as log:
        run([s.engine, s.model / "transformer", out / "case.safetensors", out], stderr=log)
    run(tool(s, "decode", s.model, out / "latents.f32", out / "zig.png", "--device", s.device, *size))
    pairs = [("ours_vs_torch_bf16", "reference.png", "zig.png")]
    if args.exact:
        run(tool(s, "exact", s.model, out, "--steps", str(args.steps), "--device", s.device))
        pairs += [("torch_bf16_vs_fp32", "exact.png", "reference.png"), ("ours_vs_fp32", "exact.png", "zig.png")]
    summary = {}
    for key, a, b in pairs:
        got = subprocess.run([str(c) for c in tool(s, "compare", out / a, out / b)], check=True, capture_output=True,
                             text=True)
        summary[key] = json.loads(got.stdout)
    steps = json.loads((out / "run.json").read_text())["teacher_forced"]
    summary["worst_step_cosine"] = {k: min(x[f"{k}_bf16"]["cosine"] for x in steps) for k in ("prompt", "negative")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


def status(s: Settings, _args) -> int:
    rows = {"model": s.model, "engine": s.engine, "venv": s.venv_python, "case tool": s.case_tool}
    print(f"{s.model_repo} @ {s.model_revision[:8]}, engine {s.tensorfold_sha[:12]}")
    for what, p in rows.items():
        print(f"{what:10} {'ok ' if p.exists() else 'MISSING'} {p}")
    return 0


def main() -> int:
    s = load()
    parser = argparse.ArgumentParser(prog="python3 -m mac", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("build")
    sub.add_parser("status")
    for name in ("generate", "check"):
        p = sub.add_parser(name)
        if name == "generate":
            p.add_argument("prompt")
            p.add_argument("-o", "--out", default="moss-image.png")
            p.add_argument("--negative", default=s.negative)
        else:
            p.add_argument("prompt_file")
            p.add_argument("--exact", action="store_true", help="also the schedule with a float32 transformer (slow)")
        p.add_argument("--width", type=int, default=s.width)
        p.add_argument("--height", type=int, default=s.height)
        p.add_argument("--steps", type=int, default=s.steps)
        p.add_argument("--seed", type=int, default=s.seed)
    args = parser.parse_args()
    if shutil.which("git") is None:
        sys.exit("git is needed")
    return {"setup": setup, "build": build, "generate": generate, "check": check, "status": status}[args.cmd](s, args)


if __name__ == "__main__":
    raise SystemExit(main())
