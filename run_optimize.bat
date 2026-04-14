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

echo [deps] Installing / verifying asrbench dependencies in 'bench'...
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

echo.
echo [mode] Default: 2-stage IAMS search (Hyperband eta=4)
echo        S1 coarse  -> 900s dataset, budget 120, eps 0.020 (noise-floor calibrated)
echo        S2 refine  -> 3600s dataset, budget 80,  eps 0.005 (warm-start from S1)
echo        Override with --single-stage or --stage[12]-duration/budget/epsilon flags.
echo.

python "%SCRIPT_DIR%optimize_matrix.py" optimize_matrix.json %*

endlocal
