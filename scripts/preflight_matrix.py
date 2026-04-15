"""Preflight validator for optimize_matrix.json.

Runs before the real IAMS matrix run to:
  1. Parse the JSON (fail fast on syntax errors)
  2. Verify every referenced space_file exists on disk
  3. Verify every dataset is known to dataset_manager._HF_SOURCE_MAP
     or declared as local_path
  4. Enforce the dataset-quality pairing rule (clean corpora → space_clean,
     noisy corpora → space_noisy)
  5. Print a readable study-by-study plan grouped by pipeline

Exit code 0 on success, 1 on any validation failure. run_optimize.bat
aborts the run when this exits non-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

# Known dataset quality labels. Sync with asrbench/data/dataset_manager.py
# _HF_SOURCE_MAP whenever you add a corpus. This is the pairing rule's
# source of truth: "clean" goes with space_clean.yaml, "noisy" with
# space_noisy.yaml. If you add a corpus whose quality varies by split
# or language, add a tuple key here.
_DATASET_QUALITY: dict[str, str] = {
    "librispeech": "clean",  # test-clean audiobook studio
    "fleurs": "clean",  # Google FLEURS multilingual studio
    "common_voice": "mixed",  # crowdsourced, variable quality
    "earnings22": "noisy",  # conference call recordings
    "mediaspeech": "noisy",  # media broadcast recordings
}

_CLEAN_SPACE = "space_clean.yaml"
_NOISY_SPACE = "space_noisy.yaml"


def _die(msg: str) -> NoReturn:
    print(f"[preflight] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_matrix(path: Path) -> dict:
    if not path.exists():
        _die(f"matrix config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"JSON parse failed: {exc}")


def _validate_study(
    study: dict, idx: int, matrix_dir: Path, errors: list[str], warnings: list[str]
) -> None:
    label = study.get("label") or f"<unlabeled #{idx}>"

    # Skip group header markers (studies with only "_group*" keys).
    if "lang" not in study or "dataset" not in study:
        return

    dataset = study["dataset"]
    space_file = study.get("space_file") or ""
    space_name = space_file.split("/")[-1].split("\\")[-1]

    # 1. space file must exist on disk
    space_path = Path(space_file)
    if not space_path.is_absolute():
        space_path = matrix_dir / space_file
    if not space_path.exists():
        errors.append(f"[{label}] space_file not found: {space_file}")
        return

    # 2. dataset must be in our known quality map OR use local_path
    quality = _DATASET_QUALITY.get(dataset)
    if quality is None and not study.get("local_path"):
        warnings.append(
            f"[{label}] dataset '{dataset}' not in _DATASET_QUALITY map — "
            f"pairing rule unchecked. Add it to preflight_matrix.py."
        )
        return

    # 3. enforce pairing rule
    if quality == "clean" and space_name == _NOISY_SPACE:
        errors.append(
            f"[{label}] PAIRING VIOLATION: dataset '{dataset}' is CLEAN but "
            f"space_file is {_NOISY_SPACE}. Clean corpora are "
            f"preprocessing-insensitive — pair with {_CLEAN_SPACE} instead. "
            f"See docs/PARAM_EXCLUSIONS.md 'Clean-Corpus Skip Rule'."
        )
    elif quality == "noisy" and space_name == _CLEAN_SPACE:
        errors.append(
            f"[{label}] PAIRING VIOLATION: dataset '{dataset}' is NOISY but "
            f"space_file is {_CLEAN_SPACE}. Noisy corpora need the full "
            f"restoration sweep — pair with {_NOISY_SPACE} instead."
        )
    elif quality == "mixed":
        warnings.append(
            f"[{label}] dataset '{dataset}' has mixed quality — pairing rule not strictly enforced."
        )


def _print_plan(matrix: dict) -> None:
    studies = [s for s in matrix.get("studies", []) if "lang" in s]
    models = matrix.get("models") or ([matrix["model"]] if "model" in matrix else [])

    print("=" * 68)
    print("  ASRbench IAMS Matrix — Preflight Plan")
    print("=" * 68)

    # Models
    if models:
        print(f"  Models ({len(models)}):")
        for m in models:
            name = m.get("name", "?")
            backend = m.get("backend", "?")
            print(f"    - {name}  [{backend}]")
    print()

    # Studies grouped by pipeline
    clean_studies = [s for s in studies if _CLEAN_SPACE in (s.get("space_file") or "")]
    noisy_studies = [s for s in studies if _NOISY_SPACE in (s.get("space_file") or "")]

    print(f"  Studies: {len(studies)}  ({len(clean_studies)} clean + {len(noisy_studies)} noisy)")
    print(f"  Per-model trials: ~{len(studies)} × budget  (Stage1 coarse + Stage2 refine)")
    print()

    def _fmt_row(s: dict) -> str:
        mode = f"batch={s['batch_size']}" if s.get("batch_size") else "sequential"
        return (
            f"    {s['label']:<34}  lang={s['lang']:<3}  "
            f"dataset={s['dataset']:<13} split={s.get('split', '?'):<6} {mode}"
        )

    if clean_studies:
        print(f"  CLEAN pipeline ({_CLEAN_SPACE}) — studio audio:")
        for s in clean_studies:
            print(_fmt_row(s))
        print()

    if noisy_studies:
        print(f"  NOISY pipeline ({_NOISY_SPACE}) — real-world degraded audio:")
        for s in noisy_studies:
            print(_fmt_row(s))
        print()

    # Highlight non-standard splits (e.g. mediaspeech/tr has only 'train')
    non_test_splits = [s for s in studies if (s.get("split") or "test") != "test"]
    if non_test_splits:
        print("  Note: non-standard splits in use (HF dataset has no 'test' split):")
        for s in non_test_splits:
            print(f"    - {s['label']}  uses split={s['split']}")
        print()

    print("=" * 68)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: preflight_matrix.py <path/to/optimize_matrix.json>", file=sys.stderr)
        return 1

    matrix_path = Path(sys.argv[1]).resolve()
    matrix_dir = matrix_path.parent
    matrix = _load_matrix(matrix_path)

    studies_raw = matrix.get("studies")
    if not isinstance(studies_raw, list) or not studies_raw:
        _die("matrix has no 'studies' list or the list is empty")
    studies: list[dict] = studies_raw

    errors: list[str] = []
    warnings: list[str] = []
    for idx, study in enumerate(studies):
        _validate_study(study, idx, matrix_dir, errors, warnings)

    _print_plan(matrix)

    for w in warnings:
        print(f"[preflight] WARN: {w}", file=sys.stderr)

    if errors:
        print(file=sys.stderr)
        for e in errors:
            print(f"[preflight] ERROR: {e}", file=sys.stderr)
        print(
            f"\n[preflight] {len(errors)} error(s), {len(warnings)} warning(s) — aborting.",
            file=sys.stderr,
        )
        return 1

    print(f"[preflight] OK — {len(studies)} studies validated ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
