"""
Run training scripts from YAML configs. Each config must define `script` (path to a
Python entrypoint); remaining keys are forwarded as `--key value` CLI arguments.

Usage:
  python runner.py path/to/run.yaml
  python runner.py path/to/dir/   # runs all *.yaml / *.yml in that dir, sorted by name
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
YAML_SUFFIXES = (".yaml", ".yml")


def _collect_yaml_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in YAML_SUFFIXES:
            raise ValueError(f"Not a YAML file: {path}")
        return [path.resolve()]
    if path.is_dir():
        files: list[Path] = []
        for ext in YAML_SUFFIXES:
            files.extend(path.glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(f"No YAML files (*.yaml, *.yml) in directory: {path}")
        return sorted({p.resolve() for p in files}, key=lambda p: p.name.casefold())
    raise FileNotFoundError(f"Path does not exist: {path}")


def _load_config(config_path: Path) -> dict:
    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping (dict): {config_path}")
    return data


def _config_to_argv(cfg: dict) -> list[str]:
    """Map flat YAML key/value pairs to subprocess argv fragments (after script path)."""
    argv: list[str] = []
    for key, value in cfg.items():
        if key == "script":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        elif isinstance(value, (list, tuple)):
            argv.extend([f"--{key}", ",".join(str(x) for x in value)])
        elif isinstance(value, dict):
            raise ValueError(
                f"Nested mapping for key {key!r} is not supported; use a flat YAML config."
            )
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


def _resolve_script(script: str, config_path: Path) -> Path:
    if not script or not isinstance(script, str):
        raise ValueError("YAML must set `script` to the Python file to run (e.g. script: src/evo/evo_2phase_trainer.py)")
    candidate = Path(script)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Script not found: {candidate} (from config {config_path})")
    return candidate


def run_one_config(config_path: Path) -> int:
    cfg = _load_config(config_path)
    script = cfg.get("script")
    script_path = _resolve_script(script, config_path)
    argv = [sys.executable, str(script_path), *_config_to_argv(cfg)]
    print(f"\n=== Running {script_path.relative_to(REPO_ROOT)} with {config_path.name} ===\n", flush=True)
    root = str(REPO_ROOT)
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    proc = subprocess.run(argv, cwd=root, env=env)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python training scripts from YAML config(s).")
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config file, or a directory containing multiple YAML files",
    )
    args = parser.parse_args()
    base = args.config.expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    paths = _collect_yaml_paths(base)
    for i, cfg_path in enumerate(paths):
        code = run_one_config(cfg_path)
        if code != 0:
            print(f"\nStopped: {cfg_path.name} exited with code {code}.", flush=True)
            sys.exit(code)
    sys.exit(0)


if __name__ == "__main__":
    main()
