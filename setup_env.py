"""
setup_env.py — interactive conda environment bootstrap for ASRbench.

What this script does:

1. Detects the ``conda`` binary on PATH (Miniconda / Anaconda / Miniforge).
2. Lists existing environments and lets the user pick one, OR create a new
   one with a chosen Python version, OR skip conda entirely and install into
   the current Python.
3. Asks which optional-dependency extras to enable (dev, faster-whisper, tr,
   preprocessing, whisper-cpp, parakeet, qwen-asr, pdf) with a sensible
   recommended default.
4. Optionally installs NVIDIA cuBLAS + cuDNN CUDA 12 runtime wheels when
   faster-whisper is selected — this is the cleanest fix for the
   ``cublas64_12.dll is not found`` error seen on fresh Windows boxes
   without a system-wide CUDA 12 install.
5. Runs ``pip install -e "path[extras]"`` inside the chosen environment via
   ``conda run``, plus the GitHub ``trnorm`` install when Turkish support is
   enabled (that package is not on PyPI).
6. Writes ``.asrbench-env`` at the repo root so ``bench.bat`` and
   ``run_optimize.bat`` pick up the chosen env automatically on every run.

Design choices:

- No third-party dependencies — stdlib only. That way the script can run
  before anything is installed in the target env.
- All conda operations go through ``conda run -n <env> <cmd>`` or
  ``conda create -n <env> ...``. We deliberately do NOT call
  ``conda activate`` in a subprocess because activation mutates shell
  state, which does not persist across ``subprocess.run`` boundaries.
- User prompts are plain ``input()`` with numbered menus and Y/N defaults.
  Defaults are shown in brackets and selected on empty input so the happy
  path is "press Enter a few times, done".
- Every subprocess call streams output live (``stdout=None``) so long pip
  installs print their normal progress. Failures abort with a clear error
  and a resumable instruction.

Usage:
    python setup_env.py           # interactive
    python setup_env.py --help    # argument reference
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENV_CONFIG_FILE = REPO_ROOT / ".asrbench-env"

# The optional-dependency groups defined in pyproject.toml.
# Keep this list in sync when pyproject is extended.
EXTRA_GROUPS: dict[str, str] = {
    "dev": "Test + lint toolchain (pytest, ruff, mypy)",
    "faster-whisper": "faster-whisper backend (CTranslate2)",
    "whisper-cpp": "whisper.cpp backend (pywhispercpp)",
    "parakeet": "NVIDIA Parakeet backend (nemo_toolkit)",
    "qwen-asr": "Qwen-Audio backend (transformers + torch)",
    "tr": "Turkish WER normalizer (trnorm, pypi)",
    "preprocessing": "Audio preprocessing (pyloudnorm, noisereduce, scipy)",
    "pdf": "Report export (weasyprint)",
}

# Preset bundles — most users pick one of these.
PRESETS: dict[str, tuple[str, list[str]]] = {
    "recommended": (
        "Recommended — dev + faster-whisper + tr + preprocessing",
        ["dev", "faster-whisper", "tr", "preprocessing"],
    ),
    "minimal": (
        "Minimal — faster-whisper only (no dev tools)",
        ["faster-whisper"],
    ),
    "full": (
        "Full — every optional backend (warning: multi-GB downloads)",
        list(EXTRA_GROUPS.keys()),
    ),
    "custom": (
        "Custom — pick individual extras",
        [],  # filled in interactively
    ),
}

# The trnorm package ships on GitHub, not PyPI. When the user picks the
# "tr" extra we chain this install so Turkish WER normalization actually
# works at runtime.
TRNORM_GIT = "git+https://github.com/ysdede/trnorm.git"

# CUDA 12 runtime wheels. Installing these two gives CTranslate2 / faster-
# whisper everything it needs to load on Windows machines that lack a
# system-wide CUDA toolkit — no PATH plumbing required.
CUDA12_WHEELS = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]


# ---------------------------------------------------------------------------
# Logging helpers — colored if the terminal supports ANSI
# ---------------------------------------------------------------------------

_ANSI = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI else text


def info(msg: str) -> None:
    print(_c("36", f"[i] {msg}"))


def success(msg: str) -> None:
    print(_c("32", f"[+] {msg}"))


def warn(msg: str) -> None:
    print(_c("33", f"[!] {msg}"))


def error(msg: str) -> None:
    print(_c("31", f"[x] {msg}"), file=sys.stderr)


def section(title: str) -> None:
    bar = "=" * 62
    print()
    print(_c("1;36", bar))
    print(_c("1;36", f"  {title}"))
    print(_c("1;36", bar))


# ---------------------------------------------------------------------------
# Conda detection + env listing
# ---------------------------------------------------------------------------


@dataclass
class CondaInfo:
    """Details about the detected conda installation."""

    binary: str  # full path to conda.exe / conda
    version: str  # "23.7.4" etc.
    base_prefix: str  # root conda install path


def detect_conda() -> CondaInfo | None:
    """Return CondaInfo if conda is on PATH, else None."""
    binary = shutil.which("conda")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, "info", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        return CondaInfo(
            binary=binary,
            version=str(data.get("conda_version", "unknown")),
            base_prefix=str(data.get("root_prefix", "")),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        warn(f"conda found at {binary} but `conda info` failed: {e}")
        return None


def list_envs(conda: CondaInfo) -> list[str]:
    """
    Return human-readable env names known to conda.

    ``conda info --json`` returns a list of env *paths* under "envs". We
    convert those to bare names by taking the last path component. The base
    environment is filtered out because installing editable dev packages
    into base is almost always a mistake.
    """
    try:
        result = subprocess.run(
            [conda.binary, "info", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        envs = data.get("envs", [])
        base = data.get("root_prefix", "")
        names: list[str] = []
        for env_path in envs:
            if env_path == base:
                continue  # skip base — not a safe target
            names.append(Path(env_path).name)
        return sorted(set(names))
    except Exception as e:  # noqa: BLE001
        warn(f"Could not list conda envs: {e}")
        return []


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def prompt_text(question: str, default: str | None = None) -> str:
    """input() with an optional default shown in brackets."""
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"  {question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("    (a non-empty answer is required)")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"  {question} {hint}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("    (answer y or n)")


def prompt_choice(question: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """
    Numbered menu. ``options`` is a list of ``(key, label)`` tuples.

    Returns the selected key. ``default`` is a 1-based index into options.
    """
    print()
    print(f"  {question}")
    for i, (_, label) in enumerate(options, start=1):
        marker = " (default)" if i == default else ""
        print(f"    {i}) {label}{marker}")
    while True:
        raw = input(f"  Choose [1-{len(options)}] [{default}]: ").strip()
        if not raw:
            return options[default - 1][0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print(f"    (enter a number between 1 and {len(options)})")


def prompt_multi_select(question: str, options: list[tuple[str, str]]) -> list[str]:
    """
    Space-or-comma separated multi-select. Empty input = none.

    Returns the list of selected keys in the original order.
    """
    print()
    print(f"  {question}")
    for i, (key, label) in enumerate(options, start=1):
        print(f"    {i}) {key:<16} {label}")
    while True:
        raw = input("  Enter numbers (e.g. 1,3,5 or 1 3 5) [none]: ").strip()
        if not raw:
            return []
        parts = raw.replace(",", " ").split()
        try:
            indices = [int(p) for p in parts]
        except ValueError:
            print("    (only numbers, separated by space or comma)")
            continue
        if any(i < 1 or i > len(options) for i in indices):
            print(f"    (numbers must be between 1 and {len(options)})")
            continue
        # Preserve the original options order + dedupe
        chosen: list[str] = []
        seen: set[str] = set()
        for i, (key, _) in enumerate(options, start=1):
            if i in indices and key not in seen:
                chosen.append(key)
                seen.add(key)
        return chosen


# ---------------------------------------------------------------------------
# Env creation + installation
# ---------------------------------------------------------------------------


def create_env(conda: CondaInfo, name: str, python_version: str) -> None:
    """Create a fresh conda env with the requested Python version."""
    info(f"Creating conda env '{name}' with Python {python_version}...")
    cmd = [
        conda.binary,
        "create",
        "-n",
        name,
        f"python={python_version}",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"conda create failed (exit code {result.returncode}). Command: {' '.join(cmd)}"
        )
    success(f"Env '{name}' created.")


def run_in_env(conda: CondaInfo | None, env: str | None, cmd: list[str]) -> None:
    """
    Execute ``cmd`` inside ``env`` via ``conda run -n``, or in the current
    Python when conda/env is None (pip fallback path).

    Output streams live so long pip installs are visible to the user.
    """
    if conda is not None and env is not None:
        full = [conda.binary, "run", "-n", env, "--live-stream", *cmd]
    else:
        full = cmd
    info(f"$ {' '.join(full)}")
    result = subprocess.run(full)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(full)}\n"
            "Fix the error above and re-run setup_env.py — it is idempotent."
        )


def build_extras_spec(extras: list[str]) -> str:
    """
    Build the ``asrbench[extras,...]`` pip extras spec.

    The project itself is installed as an editable path (``pip install -e .``)
    so local edits take effect without a reinstall. Empty ``extras`` yields
    a bare ``.`` which installs the base package only.
    """
    if not extras:
        return "."
    joined = ",".join(extras)
    return f".[{joined}]"


def install_asrbench(
    conda: CondaInfo | None,
    env: str | None,
    extras: list[str],
) -> None:
    """Run pip install -e .[extras] inside the target env."""
    spec = build_extras_spec(extras)
    info(f"Installing asrbench with extras: {extras or '(none)'}")
    run_in_env(conda, env, ["pip", "install", "-e", spec])


def install_trnorm(conda: CondaInfo | None, env: str | None) -> None:
    """Install trnorm from GitHub — not on PyPI."""
    info("Installing trnorm from GitHub (Turkish WER normalizer)...")
    run_in_env(conda, env, ["pip", "install", TRNORM_GIT])


def install_cuda12_wheels(conda: CondaInfo | None, env: str | None) -> None:
    """Install NVIDIA CUDA 12 runtime wheels (cuBLAS + cuDNN)."""
    info("Installing NVIDIA CUDA 12 runtime wheels (cuBLAS + cuDNN)...")
    run_in_env(conda, env, ["pip", "install", *CUDA12_WHEELS])


# ---------------------------------------------------------------------------
# Config file writing
# ---------------------------------------------------------------------------


def write_env_config(env_name: str) -> None:
    """
    Persist the chosen env name to ``.asrbench-env`` at the repo root.

    Format is a minimal KEY=VALUE file so the bat wrappers can parse it
    with a ``for /f`` loop and no external tools. Lines starting with ``#``
    are comments.
    """
    content = (
        "# Generated by setup_env.py — tells bench.bat and run_optimize.bat\n"
        "# which conda env to activate. Delete this file to return to the\n"
        "# legacy hardcoded default (env name 'bench').\n"
        f"ASRBENCH_CONDA_ENV={env_name}\n"
    )
    ENV_CONFIG_FILE.write_text(content, encoding="utf-8")
    success(f"Wrote {ENV_CONFIG_FILE.name} — env='{env_name}'")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def choose_env(conda: CondaInfo) -> tuple[str, bool]:
    """
    Walk the user through env selection.

    Returns ``(env_name, created_new)``. ``created_new`` is True when a
    fresh env was created; callers use it to decide whether to default
    extras + CUDA wheels to "yes" for the install step.
    """
    existing = list_envs(conda)
    print()
    info(f"Detected conda {conda.version} at {conda.binary}")
    if existing:
        info(f"Existing environments: {', '.join(existing)}")
    else:
        info("No existing user environments besides base.")

    # Build a numbered menu: each existing env, then "create new"
    options: list[tuple[str, str]] = []
    for name in existing:
        options.append((f"use:{name}", f"Use existing env: {name}"))
    options.append(("create", "Create a new env"))
    options.append(("skip", "Skip conda — install into the current Python interpreter"))

    # Default: first existing "bench"-like env, else "create"
    default_idx = 1
    for i, (key, _) in enumerate(options, start=1):
        if key == "use:bench":
            default_idx = i
            break
    else:
        # no bench-named env → default to "create"
        for i, (key, _) in enumerate(options, start=1):
            if key == "create":
                default_idx = i
                break

    choice = prompt_choice(
        "Which environment should ASRbench live in?",
        options,
        default=default_idx,
    )

    if choice == "skip":
        warn(
            "You chose to skip conda. Packages will install into the Python "
            "currently running this script. That is fine for one-shot use "
            "but NOT recommended for isolation."
        )
        return ("", False)  # empty env name means "current Python"

    if choice == "create":
        name = prompt_text("New env name", default="asrbench")
        py_version = prompt_text("Python version", default="3.11")
        create_env(conda, name, py_version)
        return (name, True)

    # "use:<name>"
    env_name = choice.split(":", 1)[1]
    return (env_name, False)


def choose_extras(created_new: bool) -> list[str]:
    """Ask which optional extras to enable. Defaults to 'recommended'."""
    section("Optional extras")
    preset_options: list[tuple[str, str]] = [(key, label) for key, (label, _) in PRESETS.items()]
    choice = prompt_choice("Which extras bundle?", preset_options, default=1)

    if choice == "custom":
        options: list[tuple[str, str]] = list(EXTRA_GROUPS.items())
        return prompt_multi_select("Select one or more extras:", options)

    _, extras = PRESETS[choice]
    _ = created_new  # reserved for future heuristics
    return list(extras)


def choose_cuda_wheels(extras: list[str]) -> bool:
    """
    Offer to install NVIDIA CUDA 12 runtime wheels.

    Only prompted when faster-whisper is part of the install, since it is
    the backend that actually needs cuBLAS at runtime. Default is "yes"
    because the resulting install is self-contained and avoids the
    ``cublas64_12.dll not found`` class of errors.
    """
    if "faster-whisper" not in extras:
        return False
    section("GPU runtime (CUDA 12)")
    print(
        "  faster-whisper (via CTranslate2) needs cuBLAS + cuDNN at runtime.\n"
        "  The simplest fix on Windows is to pip-install the NVIDIA wheels:\n"
        "    - nvidia-cublas-cu12\n"
        "    - nvidia-cudnn-cu12\n"
        "  These are ~600 MB combined but make the install self-contained."
    )
    return prompt_yes_no("Install CUDA 12 runtime wheels now?", default=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive conda environment bootstrap for ASRbench",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip all prompts; use defaults (bench env, recommended extras, CUDA wheels on).",
    )
    parser.add_argument(
        "--env",
        metavar="NAME",
        help="Environment name to use (creates it if missing). Implies --non-interactive.",
    )
    args = parser.parse_args()

    section("ASRbench environment bootstrap")

    conda = detect_conda()
    if conda is None:
        warn(
            "No conda binary found on PATH. You can still install into the "
            "current Python interpreter, but isolation is not guaranteed."
        )
        if not prompt_yes_no("Continue with the current Python?", default=False):
            error("Aborted. Install Miniconda / Miniforge first and re-run.")
            sys.exit(1)
        env_name = ""
        created_new = False
    else:
        if args.env:
            # Non-interactive path: pick/create the requested env
            existing = list_envs(conda)
            if args.env in existing:
                env_name = args.env
                created_new = False
            else:
                create_env(conda, args.env, "3.11")
                env_name = args.env
                created_new = True
        elif args.non_interactive:
            env_name = "bench"
            existing = list_envs(conda)
            if "bench" not in existing:
                create_env(conda, "bench", "3.11")
                created_new = True
            else:
                created_new = False
        else:
            env_name, created_new = choose_env(conda)

    # Extras + CUDA
    if args.non_interactive or args.env:
        extras = PRESETS["recommended"][1]
        cuda_wheels = True
    else:
        extras = choose_extras(created_new)
        cuda_wheels = choose_cuda_wheels(extras)

    # Confirm plan
    section("Install plan")
    target = f"conda env '{env_name}'" if env_name else "current Python"
    print(f"  Target:  {target}")
    print(f"  Repo:    {REPO_ROOT}")
    print(f"  Extras:  {', '.join(extras) if extras else '(none, base only)'}")
    print(f"  trnorm:  {'yes (from GitHub)' if 'tr' in extras else 'no'}")
    print(f"  CUDA12:  {'yes (nvidia-cublas-cu12 + nvidia-cudnn-cu12)' if cuda_wheels else 'no'}")
    if not (args.non_interactive or args.env):
        if not prompt_yes_no("Proceed with this plan?", default=True):
            info("Cancelled. No changes made.")
            return

    # Execute
    section("Installing")
    effective_conda = conda if env_name else None
    effective_env = env_name if env_name else None

    try:
        install_asrbench(effective_conda, effective_env, extras)
        if "tr" in extras:
            install_trnorm(effective_conda, effective_env)
        if cuda_wheels:
            install_cuda12_wheels(effective_conda, effective_env)
    except RuntimeError as e:
        error(str(e))
        sys.exit(2)

    # Persist env choice for bat wrappers
    if env_name:
        write_env_config(env_name)
    else:
        # Clear any stale config so run_optimize.bat / bench.bat fall back
        if ENV_CONFIG_FILE.exists():
            ENV_CONFIG_FILE.unlink()
            info(f"Removed stale {ENV_CONFIG_FILE.name} (using current Python).")

    section("Done")
    success("ASRbench is ready.")
    print()
    print("  Next steps:")
    if env_name:
        print(f"    1. Open a new terminal (env config is picked up from {ENV_CONFIG_FILE.name})")
        print("    2. Run: bench.bat            (interactive single benchmark)")
        print("    3. Run: run_optimize.bat     (2-stage IAMS optimizer matrix)")
    else:
        print("    1. Run: python -m asrbench.cli.app serve")
        print("    2. Run: python optimize_matrix.py optimize_matrix.json --dry-run")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        warn("Interrupted by user. No partial install — re-run when ready.")
        sys.exit(130)
