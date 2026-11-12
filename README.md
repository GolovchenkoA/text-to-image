# Local Text-to-Image

A small Python CLI that lists verified local text-to-image models and generates an image with Hugging Face Diffusers. It downloads weights only when `generate` is selected. Generated files and model caches stay outside the source files.

## Python and dependency compatibility

The project requires exactly **Python 3.14.0** and checks that version at startup. Compatibility was checked on 2026-11-12 against PyPI release metadata: PyTorch 2.15.0 publishes CPython 3.14 wheels for Windows x64 and Linux x64/ARM64, plus Apple Silicon macOS 14+. Diffusers 0.43.0, Transformers 5.22.2, Accelerate 1.15.0, Hugging Face Hub 2.4.0, SentencePiece 0.2.3, Protobuf 7.36.2, Safetensors 0.8.0, and psutil 7.2.2 declare Python ranges that include 3.14. An Intel Mac does not have a matching PyTorch 3.14 wheel in the checked release. No Python downgrade is used.

Diffusers 0.43.0 contains the pipelines used here: `FluxPipeline`, `StableDiffusion3Pipeline`, `StableDiffusionXLPipeline`, `PixArtSigmaPipeline`, and `SanaPipeline`. Sana’s Diffusers-format model card also documents `SanaPipeline`.

## Setup

Install Python 3.14.0 first. Create a project-local environment, install PyTorch from the official selector for your machine, and then install the pinned model dependencies.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Choose one PyTorch install method. For CPU-only use, run:

```powershell
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

#### Enable CUDA on Windows

1. Install or update the NVIDIA driver, then check that Windows sees the GPU with `nvidia-smi`.
2. Activate `.venv`. If you installed the CPU-only PyTorch wheel above, remove it first:

   ```powershell
   python -m pip uninstall -y torch
   ```

3. Select Windows, Pip, Python, and the CUDA version supported by your driver at [the PyTorch selector](https://pytorch.org/get-started/locally/). The current selector lists CUDA 13.4 for stable Windows/Pip, with a Python 3.14 wheel for `torch 2.15.0`. For that option, run:

   ```powershell
   python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
   ```

   If the selector recommends a different CUDA option for your driver, use its generated command instead. The pip wheel supplies the CUDA runtime for normal inference; a separate CUDA Toolkit is generally only needed to compile CUDA extensions.

After installing either CPU or CUDA PyTorch, install the remaining dependencies. You can also verify CUDA visibility:

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('Wheel CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

`CUDA available: True` means this project will select the GPU for generation. If it is `False`, check the NVIDIA driver and confirm that the CUDA wheel was installed inside the active `.venv`.

### macOS (Apple Silicon, macOS 14+)

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install torch==2.15.0
python -m pip install -r requirements.txt
```

### Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version
```

For CUDA, select Linux, Pip, Python, and the CUDA version supported by your driver at [the PyTorch selector](https://pytorch.org/get-started/locally/) and run its install command. For CPU-only use, run:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the other dependencies:

```bash
python -m pip install -r requirements.txt
```

The CUDA package/index changes by platform and supported CUDA release. Use the current [PyTorch installation selector](https://pytorch.org/get-started/locally/) rather than guessing a CUDA wheel URL. The CLI checks `torch.cuda.is_available()` and selects CUDA when true; otherwise it selects CPU and warns that generation can be very slow. The environment check does not guarantee enough free VRAM for a given model.

To install the lightweight tests, run `python -m pip install -r requirements-dev.txt`.

## Usage

List models without importing Diffusers or downloading weights:

```powershell
python -m text2image list
```

Show a model’s description, support status, license, and hardware estimates:

```powershell
python -m text2image info flux-schnell
```

Generate an image:

```powershell
python -m text2image generate --model flux-schnell --prompt "A red fox in a snowy pine forest, cinematic lighting" --output outputs/fox.png --width 1024 --height 1024 --steps 4 --seed 42
```

`generate` is a one-shot command: it exits after saving the image, so its process-local model memory is released when the process exits. To generate several images without reloading the selected model, start an interactive session:

```powershell
python -m text2image session --model flux-schnell --output-dir outputs --width 1024 --height 1024 --steps 4
```

Enter a prompt at each `Prompt>` line. The session saves files as `outputs/image-001.png`, `image-002.png`, and so on, skipping existing names. Type `:quit` (or press Ctrl+C/Ctrl+Z+Enter) to end the session. The selected model is loaded only when the first prompt is generated, reused for later prompts, and released when the session exits.

### Model cache configuration

The project-root `config.toml` controls how long a loaded pipeline remains in memory while an interactive session is idle:

```toml
[model_cache]
enabled = true
idle_timeout_seconds = 600
```

`600` means ten minutes after the last generation. If another prompt arrives before expiry, the timer is reset and the loaded pipeline is reused. After the timeout, it is released; the next prompt reloads it. Set `enabled = false` or `idle_timeout_seconds = 0` to release the pipeline after each generated image. This is an in-process cache, not a disk cache: separate `generate` invocations cannot share loaded GPU/CPU memory, so use `session` for reuse.

Other model keys: `flux-dev`, `sd35-large`, `sd35-medium`, `sdxl`, `pixart-sigma`, and `sana`. Dimensions must be positive multiples of 8. Steps default to a model-specific starting point. The saved absolute path is printed after generation.

For FLUX.1 [dev] and Stable Diffusion 3.5, first accept the terms on the corresponding Hugging Face model page and authenticate locally with `hf auth login`. A list or details command never fetches model files. The generation command warns about the license immediately before downloading and running the selected model.

## Hardware estimates

The model details show separate minimum and recommended system RAM and GPU VRAM estimates, plus an upper configuration note. Estimates assume 1024×1024, batch 1, the entire pipeline resident on the selected device, and no quantization or CPU offload. GPU VRAM assumes FP16/BF16; system RAM assumes FP32 CPU generation and includes runtime/component overhead. They are planning estimates, not hard limits or guaranteed minimums. Actual memory changes with resolution, attention implementation, precision, pipeline version, and other programs using the GPU. The registry links each estimate to its official model card and Diffusers pipeline guide; no tested maximum GPU size is claimed. Sana’s card describes output up to 4096×4096, which is a model resolution statement rather than a hardware limit.

CPU generation is available through Diffusers but can be impractically slow for large models. Before weight loading, the CLI compares available system RAM and CUDA VRAM with the estimate and warns on likely memory pressure without blocking the attempt. Out-of-memory errors suggest reducing dimensions or choosing a smaller model.

## Model support and licenses

All seven entries use the pipeline class documented by the current model card and present in Diffusers 0.43.0. Model weights are governed by their own terms, not the project license. FLUX.1 [dev] uses a gated noncommercial license; Stable Diffusion 3.5 uses Stability AI’s gated Community License; FLUX.1 [schnell] is Apache-2.0; SDXL 1.0 and PixArt-Σ use OpenRAIL++; Sana is Apache-2.0 and its card also points to Gemma text-encoder terms. Always inspect the linked license before use.

## Tests

The tests cover registry metadata, CLI parsing/listing, and CUDA/CPU device selection. They do not load pipelines or download model weights.

```powershell
python -m pytest
```
