"""Helper for bench.bat — reads config defaults and provides dataset info.

Called from bench.bat via: python -m asrbench.cli.bench_helper <command> [args]

Commands:
    defaults    — print [bench] config values as KEY=VALUE lines
    datasets    — print available datasets for a language with duration info
"""

from __future__ import annotations

import sys

# Approximate test-split durations (hours) per dataset per language.
# Sources: HuggingFace dataset cards, measured locally.
# "?" means available but duration unknown.
_DATASET_HOURS: dict[str, dict[str, str]] = {
    "fleurs": {
        "tr": "3h",
        "en": "4h",
        "de": "3h",
        "fr": "3h",
        "es": "3h",
        "ar": "3h",
        "zh": "3h",
        "ja": "3h",
        "ko": "3h",
    },
    "librispeech": {
        "en": "5h",
    },
    "mediaspeech": {
        "tr": "10h",
        "en": "10h",
        "fr": "10h",
        "ar": "10h",
    },
    "earnings22": {
        "en": "5h",
    },
}

_DATASET_LANGS: dict[str, set[str]] = {
    "fleurs": {"tr", "en", "de", "fr", "es", "ar", "zh", "ja", "ko"},
    "librispeech": {"en"},
    "mediaspeech": {"tr", "en", "fr", "ar"},
    "earnings22": {"en"},
    # common_voice omitted — gated dataset, streaming broken as of 2026-04
}

# Datasets that only have a "train" split (no "test")
_TRAIN_ONLY: set[str] = {"mediaspeech"}


def cmd_defaults() -> None:
    """Print [bench] config defaults as KEY=VALUE lines for batch consumption."""
    from asrbench.config import get_config

    get_config.cache_clear()
    cfg = get_config()
    b = cfg.bench
    # Only emit non-empty values
    if b.lang:
        print(f"CFG_LANG={b.lang}")
    if b.dataset:
        print(f"CFG_DATASET={b.dataset}")
    if b.model:
        print(f"CFG_MODEL={b.model}")
    if b.condition:
        print(f"CFG_CONDITION={b.condition}")
    if b.max_duration_s > 0:
        print(f"CFG_MAX_DURATION_S={int(b.max_duration_s)}")


def cmd_datasets(lang: str) -> None:
    """Print available datasets for a language with duration info."""
    idx = 0
    for ds_name, langs in _DATASET_LANGS.items():
        if lang in langs:
            idx += 1
            hours = _DATASET_HOURS.get(ds_name, {}).get(lang, "?")
            split = "train" if ds_name in _TRAIN_ONLY else "test"
            # Output: IDX|NAME|HOURS|SPLIT  (pipe-delimited for easy batch parsing)
            print(f"{idx}|{ds_name}|~{hours}|{split}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m asrbench.cli.bench_helper <defaults|datasets> [args]", file=sys.stderr
        )
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "defaults":
        cmd_defaults()
    elif cmd == "datasets":
        if len(sys.argv) < 3:
            print("Usage: datasets <lang>", file=sys.stderr)
            sys.exit(1)
        cmd_datasets(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
