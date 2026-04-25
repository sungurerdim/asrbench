"""
FFmpeg preprocessing backend — byte-accurate mobile pipeline parity.

The scipy/numpy pipeline in ``pipeline.py`` is signal-theory equivalent to
the FFmpeg chain most mobile apps (including SafeScribeAI) run at capture
time, but it is NOT byte-identical: loudnorm's two-pass dynamic mode,
alimiter's look-ahead, silenceremove's hysteresis, and swresample's
interpolation all differ in subtle ways between the scipy implementation
and FFmpeg's compiled filters.

That gap does not matter for generic ASR research, but it DOES matter when
the tuned config is deployed on a device that runs FFmpeg at capture time:
the IAMS optimum found against scipy may not be the mobile optimum. To
close that loop, this module reconstructs the same parameter bundle as a
``filter_complex`` string and invokes the system ``ffmpeg`` binary via a
subprocess. One invocation per segment; buffered stdin/stdout.

Cost: ~2-5x slower per trial than scipy (subprocess overhead + ffmpeg
init). Used only when the caller explicitly opts in via the pipeline
``backend="ffmpeg"`` switch.

Availability: if ``ffmpeg`` is not on PATH, the module raises at import
time. The outer pipeline catches this and falls back to scipy with a
log warning so CI environments without FFmpeg don't break.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

import numpy as np

# Allow-list for string-valued params that could theoretically reach the
# filter chain. Each pattern is anchored and matches only characters that
# are safe inside an ffmpeg filter_complex expression — letters, digits,
# underscore, hyphen. ``format`` is the one string param wired today; the
# check stays in place so adding new filters with string options does not
# accidentally open an injection path.
_SAFE_TOKEN_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_\-]+$")

_STRING_PARAM_ALLOWLIST: dict[str, frozenset[str]] = {
    "format": frozenset({"none", "opus_64k", "opus_32k", "opus_24k"}),
}


class FFmpegNotAvailable(RuntimeError):
    """Raised when the ``ffmpeg`` binary is not on PATH."""


def _validate_string_params(params: dict[str, Any]) -> None:
    """Reject string values that could inject into an ffmpeg filter chain.

    Every numeric knob in :func:`build_filter_chain` is cast via ``int()``
    or ``float()`` which already defeats injection. The remaining string
    knobs (today: ``format``) are checked against an explicit allow-list
    here so adding a new string parameter without also extending the
    allow-list fails loudly instead of silently forwarding attacker input
    into the subprocess command line.
    """
    for key, allowed in _STRING_PARAM_ALLOWLIST.items():
        if key not in params:
            continue
        value = params[key]
        if value is None:
            continue
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"Invalid value for preprocess param {key!r}: {value!r}. "
                f"Expected one of: {sorted(allowed)}."
            )

    # Defensive sweep: any *other* string value must at least be a safe
    # token. This catches future filters that forget to extend the
    # allow-list above.
    for key, value in params.items():
        if key in _STRING_PARAM_ALLOWLIST or value is None:
            continue
        if isinstance(value, str) and not _SAFE_TOKEN_RE.fullmatch(value):
            raise ValueError(
                f"Preprocess param {key!r} carries an unsafe string value "
                f"({value!r}); only alphanumerics, underscore and hyphen are "
                "permitted for string knobs reaching the ffmpeg backend."
            )


def is_ffmpeg_available() -> bool:
    """Return True iff an ``ffmpeg`` binary is available in PATH."""
    return shutil.which("ffmpeg") is not None


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotAvailable(
            "ffmpeg binary not found on PATH. Install it "
            "(https://ffmpeg.org/download.html) or keep using the default "
            "scipy preprocessing backend."
        )
    return path


# ---------------------------------------------------------------------------
# Filter chain builder — maps preprocess.* params to FFmpeg filter fragments
# ---------------------------------------------------------------------------


def _resample_fragments(params: dict[str, Any], sr: int) -> list[str]:
    target_sr = int(params.get("sample_rate", sr) or sr)
    if target_sr == sr:
        return []
    # Upsample back to sr so the rest of the chain matches the caller's
    # expected output rate. swresample will round-trip.
    return [f"aresample={target_sr}", f"aresample={sr}"]


def _notch_fragment(params: dict[str, Any], _sr: int) -> str | None:
    hz = int(params.get("notch_hz", 0) or 0)
    return f"bandreject=f={hz}:width_type=q:w=30" if hz > 0 else None


def _highpass_fragment(params: dict[str, Any], _sr: int) -> str | None:
    hz = int(params.get("highpass_hz", 0) or 0)
    return f"highpass=f={hz}" if hz > 0 else None


def _lowpass_fragment(params: dict[str, Any], _sr: int) -> str | None:
    hz = int(params.get("lowpass_hz", 0) or 0)
    return f"lowpass=f={hz}" if hz > 0 else None


def _loudnorm_fragment(params: dict[str, Any], _sr: int) -> str | None:
    # ``tp=-1.5`` (true-peak ceiling) is hardcoded. It is a functional
    # knob (±2.4 pp in the SafeScribeAI phase1 sweep) but kept pinned
    # to the EBU R128 streaming default because tuning it adds a
    # dimension with negligible sensitivity on balanced corpora.
    lufs_target = params.get("lufs_target")
    if lufs_target is None:
        return None
    lra = float(params.get("lufs_lra", 7.0))
    linear = "true" if params.get("loudnorm_linear", False) else "false"
    return f"loudnorm=I={float(lufs_target)}:LRA={lra}:tp=-1.5:linear={linear}"


def _compressor_fragment(params: dict[str, Any], _sr: int) -> str | None:
    ratio = float(params.get("drc_ratio", 1.0) or 1.0)
    return f"acompressor=ratio={ratio}:threshold=-20dB" if ratio > 1.0 else None


def _limiter_fragment(params: dict[str, Any], _sr: int) -> str | None:
    # ``attack=5`` and ``release=50`` are intentionally hardcoded. The
    # SafeScribeAI phase1 WER sweep (wer_results_log.jsonl, 2026-04-14)
    # tested ``release_ms ∈ {50, 150}`` across 6 runs × 3 datasets and
    # every trial landed at Δwer_pp = 0.000 (bit-identical to baseline) —
    # the release knob has zero measurable impact on WER in this regime.
    ceiling_db = float(params.get("limiter_ceiling_db", 0.0) or 0.0)
    if ceiling_db >= 0.0:
        return None
    limit_linear = 10.0 ** (ceiling_db / 20.0)
    return f"alimiter=limit={limit_linear:.4f}:attack=5:release=50"


def _noise_fragment(params: dict[str, Any], _sr: int) -> str | None:
    strength = float(params.get("noise_reduce", 0.0) or 0.0)
    if strength <= 0.0:
        return None
    nf_db = -30.0 * strength  # map [0..0.8] → [0..-24] dB
    return f"afftdn=nf={nf_db:.1f}"


def _preemph_fragment(params: dict[str, Any], _sr: int) -> str | None:
    coef = float(params.get("preemph_coef", 0.0) or 0.0)
    return f"aeval=val(0)-{coef}*val(-1)" if coef > 0.0 else None


def _vad_fragment(params: dict[str, Any], _sr: int) -> str | None:
    if not params.get("vad_trim", False):
        return None
    threshold_db = float(params.get("silence_threshold_db", -40.0))
    min_duration = float(params.get("silence_min_duration_s", 0.5))
    return (
        "silenceremove="
        f"start_periods=1:start_duration={min_duration}:"
        f"start_threshold={threshold_db}dB:"
        f"stop_periods=-1:stop_duration={min_duration}:"
        f"stop_threshold={threshold_db}dB"
    )


# Dispatch table — order mirrors ``PreprocessingPipeline.apply`` so scipy
# and ffmpeg backends produce equivalent outputs.
_SINGLE_FRAGMENT_BUILDERS: tuple[Any, ...] = (
    _notch_fragment,
    _highpass_fragment,
    _lowpass_fragment,
    _loudnorm_fragment,
    _compressor_fragment,
    _limiter_fragment,
    _noise_fragment,
    _preemph_fragment,
    _vad_fragment,
)


def build_filter_chain(params: dict[str, Any], *, sr: int) -> str:
    """
    Build an ``-af`` filter chain string from the preprocessing params dict.

    The order mirrors ``PreprocessingPipeline.apply`` so scipy and ffmpeg
    backends produce equivalent results on the same inputs (modulo the
    byte-level drift the whole module exists to eliminate).

    Missing / default params omit the corresponding filter. When no filter
    is needed the function returns an empty string; callers should skip
    the subprocess entirely in that case.
    """
    _validate_string_params(params)

    # Codec simulation is handled outside ffmpeg; caller pre-encodes via
    # simulate_codec because FFmpeg's own opus roundtrip would require an
    # extra encode/decode cycle that the scipy path does not model.
    filters: list[str] = list(_resample_fragments(params, sr))
    for builder in _SINGLE_FRAGMENT_BUILDERS:
        fragment = builder(params, sr)
        if fragment is not None:
            filters.append(fragment)

    return ",".join(filters)


# ---------------------------------------------------------------------------
# Subprocess execution — float32 mono in, float32 mono out
# ---------------------------------------------------------------------------


def run_ffmpeg_chain(
    audio: np.ndarray,
    filter_chain: str,
    *,
    sr: int,
) -> np.ndarray:
    """
    Feed *audio* through a compiled FFmpeg filter chain and return the result.

    Both stdin and stdout carry raw float32 little-endian mono samples at
    ``sr`` Hz. No header, no container — FFmpeg's ``f32le`` format lets us
    skip WAV encoding/decoding entirely.

    Empty ``filter_chain`` short-circuits to returning ``audio`` unchanged.
    """
    if not filter_chain:
        return audio

    ffmpeg_bin = _require_ffmpeg()

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-i",
        "pipe:0",
        "-af",
        filter_chain,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "pipe:1",
    ]

    payload = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    proc = subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg filter chain failed (rc={proc.returncode}): {stderr.strip()}\n"
            f"  chain: {filter_chain}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pipeline facade — mirrors PreprocessingPipeline.apply() but uses ffmpeg
# ---------------------------------------------------------------------------


class FFmpegPreprocessingPipeline:
    """
    FFmpeg-backed preprocessing orchestrator.

    Exposes the same ``apply(audio, params, sr)`` signature as
    ``PreprocessingPipeline`` so the benchmark engine can swap backends
    without touching the calling code.
    """

    @staticmethod
    def apply(
        audio: np.ndarray,
        params: dict[str, Any],
        sr: int = 16_000,
    ) -> np.ndarray:
        """Run the full preprocessing chain via the ffmpeg binary."""
        # Codec simulation stays on the scipy path: it's a logical step
        # (emulating a lossy encode/decode round-trip) that does not map
        # to a pre-chain ffmpeg filter.
        fmt = params.get("format", "none")
        if fmt != "none":
            from asrbench.preprocessing.codec import simulate_codec

            audio = simulate_codec(audio, fmt, sr)

        chain = build_filter_chain(params, sr=sr)
        if not chain:
            return audio
        return run_ffmpeg_chain(audio, chain, sr=sr)
