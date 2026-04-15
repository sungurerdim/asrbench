@echo off
setlocal

set "HF_HOME=D:\models\huggingface"
set "HF_HUB=D:\models\huggingface\hub"
set "SCRIPT_DIR=%~dp0"

:: ----------------------------------------------------------------------
:: Resolve conda env from .asrbench-env (written by setup_env.py) or fall
:: back to the legacy hardcoded 'bench' name. Format: ASRBENCH_CONDA_ENV=<name>
:: Lines starting with # are comments.
:: ----------------------------------------------------------------------
set "ASRBENCH_CONDA_ENV=bench"
if exist "%SCRIPT_DIR%.asrbench-env" (
    for /f "usebackq tokens=1,2 delims==" %%A in ("%SCRIPT_DIR%.asrbench-env") do (
        if /i "%%A"=="ASRBENCH_CONDA_ENV" set "ASRBENCH_CONDA_ENV=%%B"
    )
)

call conda activate %ASRBENCH_CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Conda environment '%ASRBENCH_CONDA_ENV%' not found.
    echo         Run setup.bat to create or select an environment.
    exit /b 1
)

:: Dependency fast path. Editable installs propagate source edits without
:: a reinstall, so the 5-10s "pip install -e" overhead per run is wasted
:: unless pyproject.toml changed or this is a first-time setup. Force a
:: refresh by setting ASRBENCH_REFRESH_DEPS=1 before invoking this script.
python -c "import asrbench" 2>nul
if errorlevel 1 set "ASRBENCH_REFRESH_DEPS=1"
if defined ASRBENCH_REFRESH_DEPS (
    echo [deps] Installing / refreshing asrbench dependencies in '%ASRBENCH_CONDA_ENV%'...
    pip install -e "%SCRIPT_DIR%.[faster-whisper,preprocessing]"
    if errorlevel 1 (
        echo [ERROR] pip install failed -- see output above.
        exit /b 1
    )
    pip show trnorm >nul 2>&1 || pip install "git+https://github.com/ysdede/trnorm.git"
    if errorlevel 1 (
        echo [ERROR] trnorm install failed -- Turkish WER normalization will be degraded.
    )
    echo [deps] OK
) else (
    echo [deps] asrbench already installed -- skipping ^(SET ASRBENCH_REFRESH_DEPS=1 to force^)
)

echo.
echo [mode] Default: 2-stage IAMS search via /optimize/two-stage (library-level)
echo        S1 coarse  -^> 900s  dataset, budget AUTO, eps AUTO  (sized per space)
echo        S2 refine  -^> 2400s dataset, warm-start from S1's screening
echo.
echo [features] Always-on (Faz 1-5):
echo        - Backend-aware filter: faster-whisper batched no-ops auto-excluded
echo        - Quadratic refinement: analytic minimum probe after golden section
echo        - Adaptive L2 patience: streak scales with remaining budget
echo        - Prior-informed screening: high-leverage params probed first
echo        - Per-stage auto budget + epsilon (library-sized from space + duration)
echo        - Tier 1/2 preproc: silence threshold/duration + loudnorm LRA/linear
echo.
echo [features] Opt-in flags:
echo        --preproc-backend=ffmpeg    byte-accurate mobile pipeline parity
echo                                    (2-5x slower per trial, needs FFmpeg on PATH)
echo        --global-config             one IAMS study across all datasets via
echo                                    MultiDatasetTrialExecutor (single preset
echo                                    for fleet deployment)
echo        --global-config-weights=X   duration (default) ^| uniform
echo        (edit optimize_matrix.json optimizer block for "use_multifidelity": true
echo         -- 25-40%% wall-clock savings, risk on length-sorted datasets)
echo.
echo [overrides] --single-stage  --stage[12]-duration/budget/epsilon
echo.

:: ----------------------------------------------------------------------
:: Preflight: validate matrix + print plan.
::
:: Parses optimize_matrix.json, verifies every referenced space_file
:: exists, and prints a study-by-study summary grouped by
:: clean/noisy pipeline. Fails fast if the matrix is structurally
:: broken so we do not waste model-load time on a bad config.
:: ----------------------------------------------------------------------
python "%SCRIPT_DIR%scripts\preflight_matrix.py" "%SCRIPT_DIR%optimize_matrix.json"
if errorlevel 1 (
    echo [ERROR] Matrix preflight failed -- fix optimize_matrix.json and retry.
    exit /b 1
)
echo.

python "%SCRIPT_DIR%optimize_matrix.py" optimize_matrix.json %*

endlocal
