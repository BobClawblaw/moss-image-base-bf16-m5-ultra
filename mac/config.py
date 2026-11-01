"""Settings: recipe.toml's defaults, local.toml's overrides for this Mac, then environment variables in upper case."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    checkpoint: str = "base"
    base_repo: str = ""
    base_revision: str = ""
    base_dir: str = ""
    sft_repo: str = ""
    sft_revision: str = ""
    sft_dir: str = ""
    tensorfold_src: str = ""
    tensorfold_sha: str = ""
    zig: str = "zig"
    zig_cpu: str = "native"
    binary: str = ""
    tf_cache: str = "~/.cache/moss-image"
    python: str = "python3"
    width: int = 1024
    height: int = 1024
    steps: int = 50
    seed: int = 42
    negative: str = ""
    device: str = "mps"

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    def pick(self, what: str) -> str:
        """The chosen checkpoint's repo, revision or dir."""
        if self.checkpoint not in ("base", "sft"):
            raise SystemExit(f"checkpoint must be base or sft, not {self.checkpoint!r}")
        return getattr(self, f"{self.checkpoint}_{what}")

    @property
    def model_repo(self) -> str:
        return self.pick("repo")

    @property
    def model_revision(self) -> str:
        return self.pick("revision")

    @property
    def model(self) -> Path:
        return self.path(self.pick("dir"))

    @property
    def cache(self) -> Path:
        return self.path(self.tf_cache)

    @property
    def worktree(self) -> Path:
        return self.cache / "engine" / self.tensorfold_sha[:12]

    @property
    def engine(self) -> Path:
        """tf-moss-image: `binary` when set, else the pinned worktree's build."""
        return self.path(self.binary) if self.binary else self.worktree / "zig-out" / "bin" / "tf-moss-image"

    @property
    def case_tool(self) -> Path:
        return self.worktree / "tools" / "zig" / "moss_image_case.py"

    @property
    def venv_python(self) -> Path:
        return self.cache / "venv" / "bin" / "python"


def load() -> Settings:
    values: dict = {}
    for name in ("recipe.toml", "local.toml"):
        f = ROOT / name
        if f.is_file():
            for section in tomllib.loads(f.read_text()).values():
                values.update(section)
    s = Settings()
    for f in fields(Settings):
        if f.name in values:
            setattr(s, f.name, values[f.name])
        elif f.name == "base_dir" and "model_dir" in values:  # local.toml before the SFT checkpoint was added
            s.base_dir = values["model_dir"]
        env = os.environ.get(f.name.upper())
        if env is not None:
            setattr(s, f.name, type(getattr(s, f.name))(env))
    return s
