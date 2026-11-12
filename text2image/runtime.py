"""Model loading and image generation; imported only for the generate command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .model_cache import PipelineCache
from .registry import ModelSpec


class UserFacingError(RuntimeError):
    """An actionable error suitable for a concise CLI message."""


def check_python_version(version_info: Any = sys.version_info) -> None:
    actual = tuple(version_info[:3])
    if actual != (3, 14, 0):
        version = ".".join(str(part) for part in actual)
        raise UserFacingError(
            f"Python 3.14.0 is required; detected {version}. Activate this project's .venv."
        )


def select_device(torch_module: Any) -> str:
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _pipeline_type(diffusers: Any, model: ModelSpec) -> Any:
    pipeline_type = getattr(diffusers, model.pipeline_class, None)
    if pipeline_type is None:
        raise UserFacingError(
            f"{model.name} is marked supported with Diffusers 0.43.0, but "
            f"{model.pipeline_class} is missing. Install the pinned requirements."
        )
    return pipeline_type


def _show_memory_warning(model: ModelSpec, device: str, torch_module: Any) -> None:
    try:
        import psutil

        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        if available_ram_gb < model.hardware.system_ram_min_gb:
            print(
                f"WARNING: {available_ram_gb:.1f} GB system RAM is available; "
                f"the planning estimate for {model.name} starts at "
                f"{model.hardware.system_ram_min_gb} GB. Loading may fail."
            )
    except ImportError:
        print("WARNING: Install psutil to check available system RAM before loading.")

    if device == "cuda":
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            total_gb = total_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            if total_gb < model.hardware.gpu_vram_min_gb:
                print(
                    f"WARNING: CUDA has {total_gb:.1f} GB total VRAM; the planning "
                    f"estimate for {model.name} starts at "
                    f"{model.hardware.gpu_vram_min_gb} GB. Loading may fail."
                )
            elif free_gb < model.hardware.gpu_vram_min_gb:
                print(
                    f"WARNING: Only {free_gb:.1f} GB CUDA VRAM is currently free; "
                    "other applications may leave too little memory."
                )
        except (AttributeError, RuntimeError):
            print("WARNING: Could not read free CUDA VRAM; generation may run out of memory.")
    else:
        print("WARNING: CUDA is unavailable; CPU generation can be extremely slow.")


def generate_image(
    model: ModelSpec,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    steps: int,
    seed: int | None,
    cache: PipelineCache | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise UserFacingError(
            "PyTorch is missing. Install a Python 3.14 wheel using the official "
            "PyTorch selector, then install requirements.txt."
        ) from exc

    device = select_device(torch)
    print(f"Device: {device.upper()}")
    _show_memory_warning(model, device, torch)
    print(f"License notice: {model.license_notice}")
    print(f"License details: {model.license_url}")

    try:
        import diffusers
    except ImportError as exc:
        raise UserFacingError(
            "Diffusers is missing. Install the pinned dependencies with "
            "python -m pip install -r requirements.txt."
        ) from exc

    pipeline_type = _pipeline_type(diffusers, model)
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def load_pipeline() -> Any:
        print(f"Loading only {model.name} ({model.model_id})...")
        pipeline = pipeline_type.from_pretrained(model.model_id, torch_dtype=dtype)
        if model.key == "sana" and dtype == torch.float16:
            # Sana's model card recommends an FP32 or BF16 VAE to avoid VAE
            # precision issues on CUDA devices without BF16 support.
            pipeline.vae.to(dtype=torch.float32)
        return pipeline.to(device)

    def release_device_memory() -> None:
        if device == "cuda":
            torch.cuda.empty_cache()

    cache_key = f"{model.key}:{device}:{dtype}"
    try:
        if cache is None:
            pipeline = load_pipeline()
            loaded_from_cache = False
        else:
            pipeline, loaded_from_cache = cache.get_or_load(
                cache_key,
                load_pipeline,
                cleanup=release_device_memory,
            )
        if loaded_from_cache:
            print(f"Reusing loaded {model.name} pipeline from the in-memory cache.")
    except ImportError as exc:
        raise UserFacingError(
            "A model runtime dependency is missing. Reinstall the pinned "
            "requirements in the project .venv."
        ) from exc
    except (OSError, EnvironmentError) as exc:
        message = str(exc).lower()
        if any(marker in message for marker in ("gated", "401", "403", "access", "token")):
            detail = "Accept the model terms on Hugging Face and authenticate with `hf auth login`."
        else:
            detail = "Check your network connection, model access, and free disk space."
        raise UserFacingError(f"Could not download/load {model.name}. {detail}") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise UserFacingError(
                "Not enough GPU/system memory to load this model. Close other applications, "
                "use a smaller model, or use a machine with more memory."
            ) from exc
        raise
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in ("gated", "401", "403", "access", "token")):
            detail = "Accept the model terms on Hugging Face and authenticate with `hf auth login`."
        else:
            detail = "Check your network connection, model access, and free disk space."
        raise UserFacingError(f"Could not download/load {model.name}. {detail}") from exc

    generator = None
    try:
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        result = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        )
        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise UserFacingError(
                "Generation ran out of memory. Try smaller dimensions or fewer steps, "
                "or use a machine with more memory."
            ) from exc
        raise
    finally:
        if cache is not None:
            cache.mark_idle()

    return output_path.resolve()
