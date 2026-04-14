@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: ASRbench Benchmark Runner
::
:: Interactive (tek run):   bench.bat
:: Matrix (toplu test):     bench.bat matrix.json
::                          bench.bat matrix.json --csv results.csv
::                          bench.bat matrix.json --dry-run
::
:: Config: ~/.asrbench/config.toml [bench] section for defaults
:: Models: D:\models\huggingface\hub (auto-detected local snapshots)
:: ============================================================================

set "HF_HOME=D:\models\huggingface"
set "HF_HUB=D:\models\huggingface\hub"
set "SCRIPT_DIR=%~dp0"

echo.
echo  ========================================
echo       ASRbench Benchmark Runner
echo  ========================================
echo.

:: ----------------------------------------------------------------------
:: Resolve conda env from .asrbench-env (written by setup_env.py) or fall
:: back to the legacy hardcoded 'bench' name.
:: ----------------------------------------------------------------------
set "ASRBENCH_CONDA_ENV=bench"
if exist "%SCRIPT_DIR%.asrbench-env" (
    for /f "usebackq tokens=1,2 delims==" %%A in ("%SCRIPT_DIR%.asrbench-env") do (
        if /i "%%A"=="ASRBENCH_CONDA_ENV" set "ASRBENCH_CONDA_ENV=%%B"
    )
)

:: Activate conda environment
call conda activate !ASRBENCH_CONDA_ENV!
if errorlevel 1 (
    echo [ERROR] Conda environment '!ASRBENCH_CONDA_ENV!' not found.
    echo         Run setup.bat to create or select an environment.
    goto :done
)

:: --------------------------------------------------------------------------
:: Config-driven mode: bench.bat <config.json> [--csv file] [--dry-run]
::   JSON "runs"    key → matrix_bench.py      (brute-force grid)
::   JSON "studies" key → optimize_matrix.py   (IAMS optimizer)
:: --------------------------------------------------------------------------
if not "%~1"=="" (
    if not exist "%~1" (
        echo [ERROR] File not found: %~1
        echo  Usage: bench.bat [config.json] [--csv results.csv] [--dry-run]
        goto :done
    )

    :: Detect mode by inspecting the JSON key ("studies" = optimizer, "runs" = matrix)
    set "_MODE=matrix"
    findstr /l "studies" "%~1" >nul 2>&1
    if not errorlevel 1 set "_MODE=optimize"

    if "!_MODE!"=="optimize" (
        echo  Mode: Optimizer  [%~1]
        echo.
        python "%~dp0optimize_matrix.py" %*
        goto :done
    )

    if "!_MODE!"=="matrix" (
        echo  Mode: Matrix  [%~1]
        echo.
        python "%~dp0matrix_bench.py" %*
        goto :done
    )

    echo [ERROR] Cannot determine mode from %~1
    echo  Expected a JSON with a "studies" key (optimizer) or "runs" key (matrix).
    goto :done
)

:: --------------------------------------------------------------------------
:: Load defaults from ~/.asrbench/config.toml [bench]
:: --------------------------------------------------------------------------
set "CFG_LANG="
set "CFG_DATASET="
set "CFG_MODEL="
set "CFG_CONDITION="
set "CFG_MAX_DURATION_S="

for /f "delims=" %%L in ('python -m asrbench.cli.bench_helper defaults 2^>nul') do (
    set "%%L"
)

if defined CFG_LANG echo  Config defaults: lang=!CFG_LANG! dataset=!CFG_DATASET! model=!CFG_MODEL! condition=!CFG_CONDITION!
if defined CFG_LANG echo.

:: --------------------------------------------------------------------------
:: 1. Language selection
:: --------------------------------------------------------------------------
if defined CFG_LANG (
    set "BLANG=!CFG_LANG!"
    call :resolve_lang_name !BLANG!
    echo  Language: !BLANG_NAME! [!BLANG!] (from config^)
    echo.
    goto :pick_dataset
)

echo  Available languages:
echo    1^) tr - Turkish
echo    2^) en - English
echo    3^) de - German
echo    4^) fr - French
echo    5^) es - Spanish
echo    6^) ar - Arabic
echo    7^) zh - Chinese
echo    8^) ja - Japanese
echo    9^) ko - Korean
echo.
set /p "LANG_CHOICE=  Select language [1-9] (default: 1): "
if "!LANG_CHOICE!"=="" set "LANG_CHOICE=1"

if "!LANG_CHOICE!"=="1" set "BLANG=tr"
if "!LANG_CHOICE!"=="2" set "BLANG=en"
if "!LANG_CHOICE!"=="3" set "BLANG=de"
if "!LANG_CHOICE!"=="4" set "BLANG=fr"
if "!LANG_CHOICE!"=="5" set "BLANG=es"
if "!LANG_CHOICE!"=="6" set "BLANG=ar"
if "!LANG_CHOICE!"=="7" set "BLANG=zh"
if "!LANG_CHOICE!"=="8" set "BLANG=ja"
if "!LANG_CHOICE!"=="9" set "BLANG=ko"

if not defined BLANG (
    echo [ERROR] Invalid selection: !LANG_CHOICE!
    goto :done
)
call :resolve_lang_name !BLANG!
echo.
echo  Selected: !BLANG_NAME! [!BLANG!]
echo.

:: --------------------------------------------------------------------------
:: 2. Dataset selection — shows available data size per language
:: --------------------------------------------------------------------------
:pick_dataset

:: If config has a default dataset, use it directly
if defined CFG_DATASET (
    set "DATASET=!CFG_DATASET!"
    :: mediaspeech only has train split
    if "!DATASET!"=="mediaspeech" (set "DATASET_SPLIT=train") else (set "DATASET_SPLIT=test")
    echo  Dataset: !DATASET! [!DATASET_SPLIT! split] (from config^)
    echo.
    goto :pick_model
)

echo  Available datasets for !BLANG_NAME!:
echo.

:: Query helper for dataset list with durations and splits
set "DS_COUNT=0"
for /f "tokens=1,2,3,4 delims=|" %%a in ('python -m asrbench.cli.bench_helper datasets !BLANG! 2^>nul') do (
    set /a DS_COUNT+=1
    set "DS_!DS_COUNT!=%%b"
    set "DS_SPLIT_!DS_COUNT!=%%d"
    echo    %%a^) %%b    %%c data [%%d split]
)

if "!DS_COUNT!"=="0" (
    echo    No datasets available for !BLANG!
    goto :done
)

echo.
echo  (Cached datasets reused automatically -- no re-download^)
echo.
set /p "DS_CHOICE=  Select dataset [1-!DS_COUNT!] (default: 1): "
if "!DS_CHOICE!"=="" set "DS_CHOICE=1"

set "DATASET=!DS_%DS_CHOICE%!"
set "DATASET_SPLIT=!DS_SPLIT_%DS_CHOICE%!"
if not defined DATASET_SPLIT set "DATASET_SPLIT=test"

if not defined DATASET (
    echo [ERROR] Invalid selection: !DS_CHOICE!
    goto :done
)
echo.
echo  Selected: !DATASET! [!DATASET_SPLIT! split]
echo.

:: --------------------------------------------------------------------------
:: 3. Model selection — detects local HF hub snapshots
:: --------------------------------------------------------------------------
:pick_model

if defined CFG_MODEL (
    set "MODEL_NAME=!CFG_MODEL!"
    :: Try to resolve local path for the configured model
    call :resolve_model_path "!MODEL_NAME!"
    echo  Model: !MODEL_NAME! (from config^)
    echo  Path:  !MODEL_PATH!
    echo.
    goto :pick_condition
)

echo  Available models (faster-whisper):
echo.

set "M_COUNT=0"

:: large-v3 (Systran)
set /a M_COUNT+=1
set "M_NAME_!M_COUNT!=large-v3"
call :detect_snapshot "models--Systran--faster-whisper-large-v3" "large-v3" !M_COUNT!

:: large-v3-turbo (deepdml ct2)
set /a M_COUNT+=1
set "M_NAME_!M_COUNT!=large-v3-turbo"
call :detect_snapshot "models--deepdml--faster-whisper-large-v3-turbo-ct2" "deepdml/faster-whisper-large-v3-turbo-ct2" !M_COUNT!

:: lite-large-v3 (efficient-speech acc)
set /a M_COUNT+=1
set "M_NAME_!M_COUNT!=lite-large-v3"
call :detect_snapshot "models--efficient-speech--lite-whisper-large-v3-acc" "efficient-speech/lite-whisper-large-v3-acc" !M_COUNT!

:: lite-large-v3-turbo (efficient-speech turbo acc)
set /a M_COUNT+=1
set "M_NAME_!M_COUNT!=lite-large-v3-turbo"
call :detect_snapshot "models--efficient-speech--lite-whisper-large-v3-turbo-acc" "efficient-speech/lite-whisper-large-v3-turbo-acc" !M_COUNT!

:: Generic sizes
for %%m in (medium small base tiny) do (
    set /a M_COUNT+=1
    set "M_NAME_!M_COUNT!=%%m"
    set "M_PATH_!M_COUNT!=%%m"
    echo    !M_COUNT!^) %%m [auto-download]
)

echo.
set /p "MODEL_CHOICE=  Select model [1-!M_COUNT!] (default: 1): "
if "!MODEL_CHOICE!"=="" set "MODEL_CHOICE=1"

set "MODEL_NAME=!M_NAME_%MODEL_CHOICE%!"
set "MODEL_PATH=!M_PATH_%MODEL_CHOICE%!"

if not defined MODEL_NAME (
    echo [ERROR] Invalid selection: !MODEL_CHOICE!
    goto :done
)
echo.
echo  Selected: !MODEL_NAME!
echo  Path:     !MODEL_PATH!
echo.

:: --------------------------------------------------------------------------
:: 4. Condition: clean vs noisy
:: --------------------------------------------------------------------------
:pick_condition

if defined CFG_CONDITION (
    set "CONDITION=!CFG_CONDITION!"
    echo  Condition: !CONDITION! (from config^)
    echo.
    goto :summary
)

echo  Benchmark condition:
echo    1^) clean  - Default parameters
echo    2^) noisy  - VAD filter + preprocessing
echo.
set /p "COND_CHOICE=  Select condition [1-2] (default: 1): "
if "!COND_CHOICE!"=="" set "COND_CHOICE=1"

set "CONDITION="
if "!COND_CHOICE!"=="1" set "CONDITION=clean"
if "!COND_CHOICE!"=="2" set "CONDITION=noisy"

if not defined CONDITION (
    echo [ERROR] Invalid selection: !COND_CHOICE!
    goto :done
)
echo.
echo  Selected: !CONDITION!
echo.

:: --------------------------------------------------------------------------
:: 5. Duration limit (seconds of audio to benchmark)
:: --------------------------------------------------------------------------
:pick_duration

if defined CFG_MAX_DURATION_S (
    set "MAX_DUR=!CFG_MAX_DURATION_S!"
    set /a "MAX_DUR_MIN=!MAX_DUR! / 60"
    echo  Duration: !MAX_DUR_MIN! min (from config^)
    echo.
    goto :summary
)

echo  Maximum audio duration to benchmark:
echo    1^) 10 min   (quick test^)
echo    2^) 30 min
echo    3^) 60 min   (1 hour^)
echo    4^) 180 min  (3 hours^)
echo    5^) Full dataset (no limit^)
echo.
set /p "DUR_CHOICE=  Select duration [1-5] (default: 3): "
if "!DUR_CHOICE!"=="" set "DUR_CHOICE=3"

if "!DUR_CHOICE!"=="1" set "MAX_DUR=600"
if "!DUR_CHOICE!"=="2" set "MAX_DUR=1800"
if "!DUR_CHOICE!"=="3" set "MAX_DUR=3600"
if "!DUR_CHOICE!"=="4" set "MAX_DUR=10800"
if "!DUR_CHOICE!"=="5" set "MAX_DUR=0"

if not defined MAX_DUR (
    echo [ERROR] Invalid selection: !DUR_CHOICE!
    goto :done
)

if "!MAX_DUR!"=="0" (
    echo  Selected: Full dataset
) else (
    set /a "MAX_DUR_MIN=!MAX_DUR! / 60"
    echo  Selected: !MAX_DUR_MIN! min
)
echo.

:: --------------------------------------------------------------------------
:: Summary
:: --------------------------------------------------------------------------
:summary
if not defined MAX_DUR set "MAX_DUR=0"
set "DUR_DISPLAY=full"
if not "!MAX_DUR!"=="0" set /a DUR_DISPLAY=!MAX_DUR! / 60
if not "!MAX_DUR!"=="0" set "DUR_DISPLAY=!DUR_DISPLAY! min"
echo  ----------------------------------------
echo    Benchmark Configuration
echo  ----------------------------------------
echo    Language:  !BLANG_NAME!
echo    Dataset:   !DATASET!
echo    Model:     !MODEL_NAME!
echo    Path:      !MODEL_PATH!
echo    Condition: !CONDITION!
echo    Duration:  !DUR_DISPLAY!
echo  ----------------------------------------
echo.
set /p "CONFIRM=  Proceed? [Y/n] (default: Y): "
if /i "!CONFIRM!"=="n" (
    echo  Aborted.
    goto :done
)

:: --------------------------------------------------------------------------
:: 5. Server check
:: --------------------------------------------------------------------------
set "BASE_URL=http://127.0.0.1:8765"
set "SERVER_STARTED=0"

echo.
echo  [1/5] Checking server...
curl -sf %BASE_URL%/system/health >nul 2>&1
if errorlevel 1 goto :start_server
echo        Server already running.
goto :register_model

:start_server
echo        Server not running -- starting in background...
start "" /b python -m asrbench.cli.app serve >nul 2>&1
set "SERVER_STARTED=1"
set "RETRIES=0"

:wait_server
timeout /t 2 /nobreak >nul
curl -sf %BASE_URL%/system/health >nul 2>&1
if not errorlevel 1 goto :server_ready
set /a RETRIES+=1
if !RETRIES! geq 15 (
    echo        [ERROR] Server failed to start after 30 seconds.
    goto :done
)
goto :wait_server

:server_ready
echo        Server ready.

:: --------------------------------------------------------------------------
:: 6. Register model
:: --------------------------------------------------------------------------
:register_model
echo  [2/5] Registering model !MODEL_NAME!...

set "MODEL_PATH_JSON=!MODEL_PATH:\=\\!"
set "TMPFILE=%TEMP%\asrbench_%RANDOM%.json"

curl -sf -X POST %BASE_URL%/models -H "Content-Type: application/json" -d "{\"family\":\"whisper\",\"name\":\"!MODEL_NAME!\",\"backend\":\"faster-whisper\",\"local_path\":\"!MODEL_PATH_JSON!\"}" -o "%TMPFILE%" 2>nul

if errorlevel 1 (
    echo        [ERROR] Failed to register model.
    goto :cleanup
)

for /f "delims=" %%i in ('python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['model_id'])" "%TMPFILE%" 2^>nul') do set "MODEL_ID=%%i"
del "%TMPFILE%" >nul 2>&1

if not defined MODEL_ID (
    echo        [ERROR] Could not parse model_id.
    goto :cleanup
)
echo        Model ID: !MODEL_ID!

:: --------------------------------------------------------------------------
:: 7. Load model
:: --------------------------------------------------------------------------
echo  [3/5] Loading model into memory...
curl -sf -X POST "%BASE_URL%/models/!MODEL_ID!/load" -H "Content-Type: application/json" >nul 2>&1
echo        Model loaded.

:: --------------------------------------------------------------------------
:: 8. Fetch dataset
:: --------------------------------------------------------------------------
echo  [4/5] Fetching dataset !DATASET! [!BLANG!] (max !DUR_DISPLAY!^)...

set "TMPFILE=%TEMP%\asrbench_%RANDOM%.json"
if "!MAX_DUR!"=="0" (
    curl -sf -X POST %BASE_URL%/datasets/fetch -H "Content-Type: application/json" -d "{\"source\":\"!DATASET!\",\"lang\":\"!BLANG!\",\"split\":\"!DATASET_SPLIT!\"}" -o "%TMPFILE%" 2>nul
) else (
    curl -sf -X POST %BASE_URL%/datasets/fetch -H "Content-Type: application/json" -d "{\"source\":\"!DATASET!\",\"lang\":\"!BLANG!\",\"split\":\"!DATASET_SPLIT!\",\"max_duration_s\":!MAX_DUR!}" -o "%TMPFILE%" 2>nul
)

if errorlevel 1 (
    echo        [ERROR] Failed to fetch dataset.
    goto :cleanup
)

for /f "delims=" %%i in ('python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['dataset_id'])" "%TMPFILE%" 2^>nul') do set "DATASET_ID=%%i"
del "%TMPFILE%" >nul 2>&1

if not defined DATASET_ID (
    echo        [ERROR] Could not parse dataset_id.
    goto :cleanup
)
echo        Dataset ID: !DATASET_ID!
echo        Waiting for dataset...

:wait_dataset
timeout /t 3 /nobreak >nul
set "TMPFILE=%TEMP%\asrbench_%RANDOM%.json"
curl -sf "%BASE_URL%/datasets/!DATASET_ID!" -o "%TMPFILE%" 2>nul
for /f "delims=" %%v in ('python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('verified',False))" "%TMPFILE%" 2^>nul') do set "DS_VERIFIED=%%v"
del "%TMPFILE%" >nul 2>&1
if /i "!DS_VERIFIED!"=="True" goto :dataset_ready
echo        Still loading...
goto :wait_dataset

:dataset_ready
echo        Dataset ready.

:: --------------------------------------------------------------------------
:: 9. Start benchmark run
:: --------------------------------------------------------------------------
echo  [5/5] Starting benchmark run...

if "!CONDITION!"=="clean" (
    set "RUN_PARAMS={}"
) else (
    set "RUN_PARAMS={\"vad_filter\":true,\"preprocess.noise_reduce\":true}"
)

set "TMPFILE=%TEMP%\asrbench_%RANDOM%.json"
curl -sf -X POST %BASE_URL%/runs/start -H "Content-Type: application/json" -d "{\"model_id\":\"!MODEL_ID!\",\"dataset_id\":\"!DATASET_ID!\",\"lang\":\"!BLANG!\",\"params\":!RUN_PARAMS!}" -o "%TMPFILE%" 2>nul

if errorlevel 1 (
    echo        [ERROR] Failed to start run.
    goto :cleanup
)

for /f "delims=" %%i in ('python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['run_id'])" "%TMPFILE%" 2^>nul') do set "RUN_ID=%%i"
del "%TMPFILE%" >nul 2>&1

if not defined RUN_ID (
    echo        [ERROR] Could not parse run_id.
    goto :cleanup
)
echo        Run ID: !RUN_ID!

:: --------------------------------------------------------------------------
:: Poll for completion
:: --------------------------------------------------------------------------
echo.
echo  Waiting for benchmark to complete...
echo.

:poll_run
timeout /t 5 /nobreak >nul
set "TMPFILE=%TEMP%\asrbench_%RANDOM%.json"
curl -sf "%BASE_URL%/runs/!RUN_ID!" -o "%TMPFILE%" 2>nul
for /f "delims=" %%s in ('python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['status'])" "%TMPFILE%" 2^>nul') do set "RUN_STATUS=%%s"
del "%TMPFILE%" >nul 2>&1

if "!RUN_STATUS!"=="completed" goto :run_done
if "!RUN_STATUS!"=="failed" (
    echo  [ERROR] Benchmark run failed.
    goto :cleanup
)
echo        Running...
goto :poll_run

:run_done
echo.
echo  ========================================
echo       Benchmark Complete
echo  ========================================
echo.
echo  Results:
echo  ----------------------------------------
curl -sf "%BASE_URL%/runs/!RUN_ID!" 2>nul | python -c "import sys,json;d=json.load(sys.stdin);a=d.get('aggregate') or {};lines=[('WER',a.get('wer_mean'),':.4f'),('CER',a.get('cer_mean'),':.4f'),('RTFx',a.get('rtfx_mean'),':.2f'),('Wall',a.get('wall_time_s'),':.1f'),('Words',a.get('word_count'),'')];[print(f'  {n+\":\":<10}{format(v,f)}') for n,v,f in lines if v is not None]"
echo  ----------------------------------------
echo.
echo  Full details: curl %BASE_URL%/runs/!RUN_ID!
echo  Segments:     curl %BASE_URL%/runs/!RUN_ID!/segments
echo.
goto :cleanup

:: ======================================================================
:: Subroutines
:: ======================================================================

:resolve_lang_name
:: Usage: call :resolve_lang_name <code>
:: Sets BLANG_NAME based on language code
if "%~1"=="tr" set "BLANG_NAME=Turkish"
if "%~1"=="en" set "BLANG_NAME=English"
if "%~1"=="de" set "BLANG_NAME=German"
if "%~1"=="fr" set "BLANG_NAME=French"
if "%~1"=="es" set "BLANG_NAME=Spanish"
if "%~1"=="ar" set "BLANG_NAME=Arabic"
if "%~1"=="zh" set "BLANG_NAME=Chinese"
if "%~1"=="ja" set "BLANG_NAME=Japanese"
if "%~1"=="ko" set "BLANG_NAME=Korean"
goto :eof

:resolve_model_path
:: Usage: call :resolve_model_path "model_name"
:: Sets MODEL_PATH — tries local HF snapshot first, falls back to name
set "MODEL_PATH=%~1"
set "_RM_SNAP="
if "%~1"=="large-v3" (
    if exist "%HF_HUB%\models--Systran--faster-whisper-large-v3\snapshots" (
        for /f "delims=" %%d in ('dir /b "%HF_HUB%\models--Systran--faster-whisper-large-v3\snapshots" 2^>nul') do set "_RM_SNAP=%%d"
    )
    if defined _RM_SNAP set "MODEL_PATH=%HF_HUB%\models--Systran--faster-whisper-large-v3\snapshots\!_RM_SNAP!"
)
if "%~1"=="large-v3-turbo" (
    if exist "%HF_HUB%\models--deepdml--faster-whisper-large-v3-turbo-ct2\snapshots" (
        for /f "delims=" %%d in ('dir /b "%HF_HUB%\models--deepdml--faster-whisper-large-v3-turbo-ct2\snapshots" 2^>nul') do set "_RM_SNAP=%%d"
    )
    if defined _RM_SNAP set "MODEL_PATH=%HF_HUB%\models--deepdml--faster-whisper-large-v3-turbo-ct2\snapshots\!_RM_SNAP!"
)
if "%~1"=="lite-large-v3" (
    if exist "%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-acc\snapshots" (
        for /f "delims=" %%d in ('dir /b "%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-acc\snapshots" 2^>nul') do set "_RM_SNAP=%%d"
    )
    if defined _RM_SNAP set "MODEL_PATH=%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-acc\snapshots\!_RM_SNAP!"
)
if "%~1"=="lite-large-v3-turbo" (
    if exist "%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-turbo-acc\snapshots" (
        for /f "delims=" %%d in ('dir /b "%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-turbo-acc\snapshots" 2^>nul') do set "_RM_SNAP=%%d"
    )
    if defined _RM_SNAP set "MODEL_PATH=%HF_HUB%\models--efficient-speech--lite-whisper-large-v3-turbo-acc\snapshots\!_RM_SNAP!"
)
goto :eof

:detect_snapshot
:: Usage: call :detect_snapshot "hf_dir_name" "fallback_id" index
:: Sets M_PATH_{index} and prints menu line
set "_DS_SNAP="
if exist "%HF_HUB%\%~1\snapshots" (
    for /f "delims=" %%d in ('dir /b "%HF_HUB%\%~1\snapshots" 2^>nul') do set "_DS_SNAP=%%d"
)
if defined _DS_SNAP (
    set "M_PATH_%~3=%HF_HUB%\%~1\snapshots\!_DS_SNAP!"
    echo    %~3^) !M_NAME_%~3! [LOCAL]
) else (
    set "M_PATH_%~3=%~2"
    echo    %~3^) !M_NAME_%~3! [auto-download]
)
goto :eof

:cleanup
if "!SERVER_STARTED!"=="1" (
    echo  Note: Background server is still running on port 8765.
    echo        Stop with: taskkill /f /im python.exe
)

:done
endlocal
