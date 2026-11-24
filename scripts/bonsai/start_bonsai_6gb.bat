@echo off
REM 6GB VRAM profile: High-Efficiency / Zero-OOM Launcher for 6GB GPUs.
REM Target hardware: NVIDIA GeForce GTX 1660 Ti (6 GB VRAM, Turing SM 7.5, TU116, 24 GB RAM).
REM
REM ARCHITECTURAL HARDWARE INVARIANTS ON 6GB TURING (TU116):
REM   1. ZERO Tensor Cores: TU116 lacks FP8/INT4 Tensor Cores. Kernels requiring
REM      Tensor Cores (Marlin, FlashInfer) fail or fall back to slow emulation.
REM   2. vLLM in WSL2 fails: PyTorch CUDA context (~500MB) + model weights (4.5GB)
REM      leaves <200MB for KV cache blocks, triggering instant OOM.
REM   3. llama-server (llama.cpp CUDA): Pinnacle engine. Zero PyTorch overhead,
REM      native Turing FlashAttention (-fa on), quantized KV cache (-ctk q4_0/q8_0),
REM      zero-VRAM host dynamic n-gram speculative decoding (-lcd), and native OpenAI API.
REM
REM VRAM BUDGET BREAKDOWN (6GB VRAM, GTX 1660 Ti):
REM   Qwen2.5-Coder-3B GQA KV math (36 layers, 2 KV heads, head_dim 128 = 18,432 elems/tok):
REM     - 32,768 ctx (-ctk q8_0 -ctv q8_0): ~604 MB (~612 MiB) KV -> ~3.29 GB total (>2.4 GB headroom)
REM     - 65,536 ctx (-ctk q4_0 -ctv q4_0): ~604 MB (~648 MiB) KV -> ~3.33 GB total (>2.4 GB headroom)
REM     - 131,072 ctx (-ctk q4_0 -ctv q4_0): ~1.21 GB (~1.27 GiB) KV -> ~3.96 GB total (>1.8 GB headroom)
REM   Model weights (Qwen2.5-Coder-3B Q5_K_M):                        ~2.24 GB
REM   CUDA context + activation / scratch buffers:                   ~0.45 GB
REM   ----------------------------------------------------------------
REM   Total Net GPU Footprint (64K Q4_0 KV):                         ~3.33 GB
REM   Safety Headroom:                                               ~2.67 GB (leaves room for DWM/desktop)
REM
REM Alternative Phi-4-mini (3.8B, 32 layers, 8 KV heads, head_dim 128, -c 32768 -ctk q4_0 -ctv q4_0):
REM   Model weights (Phi-4-mini-instruct-Q4_K_M):                    ~2.50 GB
REM   KV Cache (-c 32768 -ctk q4_0 -ctv q4_0):                       ~1.13 GiB
REM   CUDA context + activation buffers:                             ~0.45 GB
REM   ----------------------------------------------------------------
REM   Total Net GPU Footprint:                                       ~4.08 GB (100%% GPU resident)
REM
REM Alternative 7B Dense Model (-c 8192, Q4_K_M):
REM   Model weights (Qwen2.5-Coder-7B-Q4_K_M):                       ~4.68 GB
REM   KV Cache (-c 8192 -ctk q4_0 -ctv q4_0):                        ~0.13 GB
REM   CUDA context + activation buffers:                             ~0.45 GB
REM   ----------------------------------------------------------------
REM   Total Net GPU Footprint:                                       ~5.26 GB (100%% GPU resident)
REM
REM Stop with: stop_bonsai.bat

setlocal
cd /d "%~dp0\..\.."

REM Locate llama-server binary: check local repo path, then system PATH
set BIN=.\llama-cpp\prism-cuda\llama-server.exe
if not exist "%BIN%" (
    where llama-server.exe >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%i in ('where llama-server.exe') do (
            set "BIN=%%i"
            goto bin_found
        )
    )
)
:bin_found

REM Model auto-selection (prefers Qwen2.5-Coder-3B 64K Pareto frontier, then Phi-4-mini 32K, then 7B, then Bonsai 27B)
set MODEL=
if exist ".\models\Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf" (
    set "MODEL=.\models\Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf"
    set CTX=65536
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
) else if exist ".\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf" (
    set "MODEL=.\models\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
    set CTX=65536
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
) else if exist ".\models\Phi-4-mini-instruct-Q4_K_M.gguf" (
    set "MODEL=.\models\Phi-4-mini-instruct-Q4_K_M.gguf"
    set CTX=32768
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
) else if exist ".\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" (
    set "MODEL=.\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    set CTX=8192
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
) else if exist ".\models\Ternary-Bonsai-2-27B-PQ2_0.gguf" (
    set "MODEL=.\models\Ternary-Bonsai-2-27B-PQ2_0.gguf"
    set CTX=16384
    set CTK=q4_0
    set CTV=q4_0
    REM Partial GPU offload (38 of 64 layers to GPU, remainder in 24 GB host RAM)
    set NGL=38
)

set PORT=18020
set BATCH=2048
set UBATCH=512

if "%MODEL%"=="" (
    set "MODEL=.\models\Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf"
    set CTX=65536
    set CTK=q4_0
    set CTV=q4_0
    set NGL=99
)

if not exist "%BIN%" (
    echo [ERROR] llama-server executable not found.
    echo Please place llama-server.exe in .\llama-cpp\prism-cuda\ or install via:
    echo   winget install ggml.llamacpp
    exit /b 1
)

if not exist "%MODEL%" (
    echo [NOTICE] Model weights not found at %MODEL%
    echo Please download a supported GGUF model into .\models\
    echo Recommended for 6GB VRAM (GTX 1660 Ti):
    echo   1. Qwen2.5-Coder-3B-Instruct-Q5_K_M.gguf (2.24 GB, 64K q4_0 KV / 32K q8_0 KV, 58-68 tps)
    echo   2. Phi-4-mini-instruct-Q4_K_M.gguf (2.50 GB, 32K q4_0 KV, 55-62 tps)
    echo   3. Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf (4.68 GB, 8K q4_0 KV, 28-34 tps)
    exit /b 1
)

if not exist ".scratch" mkdir ".scratch"

echo ===================================================================
echo Starting 6GB VRAM PROFILE (Turing SM 7.5, GTX 1660 Ti)
echo Binary:         %BIN%
echo Target:         %MODEL%
echo Context:        %CTX% tokens (1 dedicated slot, -np 1)
echo KV Cache:       %CTK% / %CTV% (Quantized KV cache)
echo GPU Offload:    -ngl %NGL%
echo FlashAttention: ON (Micro-batch: %UBATCH%, Batch: %BATCH%)
echo Speculation:    Host Dynamic Lookup (-lcd .scratch\dynamic_lookup.bin)
echo Endpoint:       http://127.0.0.1:%PORT%/v1
echo ===================================================================

"%BIN%" ^
  -m "%MODEL%" ^
  -lcd .scratch\dynamic_lookup.bin ^
  --host 127.0.0.1 ^
  --port %PORT% ^
  --alias qwen2.5-coder,phi-4-mini,castor-coder ^
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
  --reasoning-format deepseek ^
  --reasoning-preserve

endlocal
