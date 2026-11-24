@echo off
REM 6GB VRAM profile: Crowned Silicon-Verified Launcher for GTX 1660 Ti (6 GB VRAM).
REM Target hardware: NVIDIA GeForce GTX 1660 Ti (Turing SM 7.5, TU116, 24 GB RAM).
REM
REM MEASURED SILICON INVARIANTS ON 6GB TURING (TU116, ZERO Tensor Cores):
REM   1. ggml-org/llama.cpp b12676 Vulkan0 (-dev Vulkan0) beats upstream CUDA 12.4
REM      release binaries by 2.1x-2.44x on prompt eval (368.4 vs 150.8 tok/s pp2048 on
REM      Bonsai-2-9B-Q2_0; 594.0 vs 280.1 tok/s on Qwen3.5-4B) because Turing TU116
REM      lacks Tensor Cores and Vulkan uses native FP16 ALU + INT dot shaders.
REM   2. Ternary-Bonsai-2-9B-Q2_0 (3:1 Gated DeltaNet + full attention) @ 64K q8_0 KV:
REM      - 100%% GPU offloaded (33/33 layers), 4149-4213 MiB VRAM used (1754-1818 MiB free)
REM      - 50.1-51.6 tok/s decode, 336-376 tok/s prefill, 290ms warm TTFT, 4.77s boot
REM      - 8/8 DGI JsonSchemaChat, 2/2 OpenAI tools, 5/5 bonsai_deep_eval (34.72s)
REM   3. Do NOT pass --spec-type ngram-mod / -lcd on Qwen3.5/Bonsai-2 GDN models:
REM      measured 0/486 draft acceptance and 2x decode slowdown (27.8 vs 53.9 tok/s).
REM
REM Stop with: stop_bonsai.bat

setlocal
cd /d "%~dp0\..\.."

REM Locate llama-server binary: prefer Vulkan build on Turing SM 7.5, then CUDA, then legacy/PATH
set BIN=
for /d %%d in (".engines\*vulkan*") do (
    if exist "%%d\llama-server.exe" (
        set "BIN=%%d\llama-server.exe"
        goto bin_found
    )
)
for /d %%d in (".engines\*cuda*") do (
    if exist "%%d\llama-server.exe" (
        set "BIN=%%d\llama-server.exe"
        goto bin_found
    )
)
for /d %%d in (".engines\*") do (
    if exist "%%d\llama-server.exe" (
        set "BIN=%%d\llama-server.exe"
        goto bin_found
    )
)
if exist ".\llama-cpp\prism-cuda\llama-server.exe" (
    set "BIN=.\llama-cpp\prism-cuda\llama-server.exe"
    goto bin_found
)
where llama-server.exe >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('where llama-server.exe') do (
        set "BIN=%%i"
        goto bin_found
    )
)
:bin_found

set DEV_ARGS=
echo "%BIN%" | findstr /I "vulkan" >nul 2>&1
if not errorlevel 1 (
    set "DEV_ARGS=-dev Vulkan0"
)

REM Model auto-selection (prefers silicon-crowned Ternary-Bonsai-2-9B-Q2_0 64K Q8 KV)
set MODEL=
set ALIAS=castor-local
if exist ".\models\Ternary-Bonsai-2-9B-Q2_0.gguf" (
    set "MODEL=.\models\Ternary-Bonsai-2-9B-Q2_0.gguf"
    set "ALIAS=Ternary-Bonsai-2-9B-Q2_0"
    set CTX=65536
    set CTK=q8_0
    set CTV=q8_0
    set NGL=99
) else if exist ".\models\Qwen3.5-4B-UD-Q4_K_XL.gguf" (
    set "MODEL=.\models\Qwen3.5-4B-UD-Q4_K_XL.gguf"
    set "ALIAS=Qwen3.5-4B-UD-Q4_K_XL"
    set CTX=65536
    set CTK=q8_0
    set CTV=q8_0
    set NGL=99
) else if exist ".\models\Qwen3.5-9B-UD-Q3_K_XL.gguf" (
    set "MODEL=.\models\Qwen3.5-9B-UD-Q3_K_XL.gguf"
    set "ALIAS=Qwen3.5-9B-UD-Q3_K_XL"
    set CTX=32768
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
) else if exist ".\models\Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf" (
    set "MODEL=.\models\Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf"
    set "ALIAS=Qwen2.5-Coder-3B-Instruct"
    set CTX=65536
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
)

set PORT=18020
set BATCH=2048
set UBATCH=512

if "%MODEL%"=="" (
    set "MODEL=.\models\Ternary-Bonsai-2-9B-Q2_0.gguf"
    set "ALIAS=Ternary-Bonsai-2-9B-Q2_0"
    set CTX=65536
    set CTK=q8_0
    set CTV=q8_0
    set NGL=99
)

if not "%CASTOR_LLAMA_CTX%"=="" set "CTX=%CASTOR_LLAMA_CTX%"
if not "%CASTOR_LLAMA_KV_TYPE%"=="" (
    set "CTK=%CASTOR_LLAMA_KV_TYPE%"
    set "CTV=%CASTOR_LLAMA_KV_TYPE%"
)

if not exist "%BIN%" (
    echo [ERROR] llama-server executable not found.
    echo Place llama-server.exe in .engines\llama-b12676-vulkan\ or .engines\^<engine^>\.
    exit /b 1
)

if not exist "%MODEL%" (
    echo [ERROR] Model weights not found at %MODEL%
    echo Place Ternary-Bonsai-2-9B-Q2_0.gguf or Qwen3.5-4B-UD-Q4_K_XL.gguf in .\models\
    exit /b 1
)

echo ===================================================================
echo Starting 6GB VRAM WINNER PROFILE (Turing SM 7.5, GTX 1660 Ti)
echo Binary:         %BIN% %DEV_ARGS%
echo Target:         %MODEL% (alias: %ALIAS%)
echo Context:        %CTX% tokens (1 dedicated slot, -np 1)
echo KV Cache:       %CTK% / %CTV% (FlashAttention ON)
echo GPU Offload:    -ngl %NGL%
echo Endpoint:       http://127.0.0.1:%PORT%/v1
echo ===================================================================

"%BIN%" ^
  -m "%MODEL%" ^
  %DEV_ARGS% ^
  --host 127.0.0.1 ^
  --port %PORT% ^
  --alias %ALIAS% ^
  --jinja ^
  -ngl %NGL% ^
  -fa on ^
  -b %BATCH% ^
  -ub %UBATCH% ^
  -c %CTX% ^
  -np 1 ^
  -ctk %CTK% ^
  -ctv %CTV% ^
  -sps 0.05 ^
  --reasoning-format deepseek

endlocal
