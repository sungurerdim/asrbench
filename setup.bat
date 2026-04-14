@echo off
setlocal

:: ============================================================================
:: ASRbench environment bootstrap
::
::   setup.bat                      interactive
::   setup.bat --non-interactive    defaults: env=bench, recommended extras
::   setup.bat --env myenv          non-interactive, use/create 'myenv'
::
:: The Python script (setup_env.py) does all the real work:
::   - detects conda
::   - prompts for env (existing / new / skip)
::   - prompts for extras (recommended / minimal / full / custom)
::   - offers CUDA 12 runtime wheels if faster-whisper is selected
::   - pip installs into the chosen env via `conda run`
::   - writes .asrbench-env so bench.bat and run_optimize.bat pick it up
::
:: This wrapper only exists so Windows users can double-click the repo's
:: setup entry point without typing a Python command.
:: ============================================================================

set "SCRIPT_DIR=%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not on PATH. Install Python 3.11+ (or Miniconda) first.
    exit /b 1
)

python "%SCRIPT_DIR%setup_env.py" %*
exit /b %errorlevel%
