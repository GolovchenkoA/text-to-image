"""Verified model metadata, independent of the ML runtime and model weights."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareEstimate:
    system_ram_min_gb: int
    system_ram_recommended_gb: int
    gpu_vram_min_gb: int
    gpu_vram_recommended_gb: int
    practical_upper: str
    assumptions: str
    sources: tuple[str, ...]
    cpu_note: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    model_id: str
    best_for: str
    main_tradeoff: str
    license_name: str
    license_url: str
    license_notice: str
    pipeline_class: str
    recommended_steps: int
    hardware: HardwareEstimate
    support_status: str


_COMMON_ASSUMPTION = (
    "Planning estimates for 1024×1024, batch 1, the full pipeline resident "
    "on the selected device, and no quantization or CPU offload. GPU VRAM "
    "assumes FP16/BF16; system RAM assumes FP32 CPU generation and includes "
    "runtime/component overhead. These are not official minimums; resolution, "
    "software versions, and options change actual use. The linked model card "
    "and Diffusers guide identify the model/pipeline; estimate bands are "
    "engineering estimates based on model scale and these assumptions."
)


def _hardware(
    ram_min: int,
    ram_recommended: int,
    vram_min: int,
    vram_recommended: int,
    *,
    model_card: str,
    pipeline_docs: str,
    upper: str = "No fixed maximum; an upper hardware configuration is not verified.",
    cpu_note: str = "CPU generation is supported by the pipeline, but is very slow and may need substantial system RAM.",
) -> HardwareEstimate:
    return HardwareEstimate(
        system_ram_min_gb=ram_min,
        system_ram_recommended_gb=ram_recommended,
        gpu_vram_min_gb=vram_min,
        gpu_vram_recommended_gb=vram_recommended,
        practical_upper=upper,
        assumptions=_COMMON_ASSUMPTION,
        sources=(model_card, pipeline_docs),
        cpu_note=cpu_note,
    )


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="flux-dev",
        name="FLUX.1 [dev]",
        model_id="black-forest-labs/FLUX.1-dev",
        best_for="High-quality photorealism, prompt following, and general-purpose generation.",
        main_tradeoff="Excellent quality, but slower and heavier; standard weights have noncommercial restrictions.",
        license_name="FLUX.1-dev Non-Commercial License (gated)",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        license_notice="Gated model. Accept the FLUX.1-dev terms on Hugging Face; standard weights are restricted to noncommercial use.",
        pipeline_class="FluxPipeline",
        recommended_steps=28,
        hardware=_hardware(
            64, 128, 40, 48,
            model_card="https://huggingface.co/black-forest-labs/FLUX.1-dev",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux",
        ),
        support_status="Supported by Diffusers 0.43.0 (FluxPipeline); Hugging Face access is gated.",
    ),
    ModelSpec(
        key="flux-schnell",
        name="FLUX.1 [schnell]",
        model_id="black-forest-labs/FLUX.1-schnell",
        best_for="Fast previews and rapid iteration.",
        main_tradeoff="Generates in very few steps, but is less refined than FLUX.1 [dev].",
        license_name="Apache License 2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md",
        license_notice="The model card lists Apache-2.0. Review the license for your use case.",
        pipeline_class="FluxPipeline",
        recommended_steps=4,
        hardware=_hardware(
            64, 128, 40, 48,
            model_card="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux",
        ),
        support_status="Supported by Diffusers 0.43.0 (FluxPipeline).",
    ),
    ModelSpec(
        key="sd35-large",
        name="Stable Diffusion 3.5 Large",
        model_id="stabilityai/stable-diffusion-3.5-large",
        best_for="Detailed compositions, typography, and strong prompt following.",
        main_tradeoff="Heavy; check Stability’s license for the intended use.",
        license_name="Stability AI Community License (gated)",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/LICENSE.md",
        license_notice="Gated model. Accept Stability AI's terms on Hugging Face and check its Community License for your intended use.",
        pipeline_class="StableDiffusion3Pipeline",
        recommended_steps=28,
        hardware=_hardware(
            64, 96, 32, 48,
            model_card="https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3",
        ),
        support_status="Supported by Diffusers 0.43.0 (StableDiffusion3Pipeline); Hugging Face access is gated.",
    ),
    ModelSpec(
        key="sd35-medium",
        name="Stable Diffusion 3.5 Medium",
        model_id="stabilityai/stable-diffusion-3.5-medium",
        best_for="Balancing image quality with more modest hardware requirements.",
        main_tradeoff="More accessible than Large, but less capable on difficult prompts.",
        license_name="Stability AI Community License (gated)",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/LICENSE.md",
        license_notice="Gated model. Accept Stability AI's terms on Hugging Face and check its Community License for your intended use.",
        pipeline_class="StableDiffusion3Pipeline",
        recommended_steps=28,
        hardware=_hardware(
            32, 48, 16, 24,
            model_card="https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3",
        ),
        support_status="Supported by Diffusers 0.43.0 (StableDiffusion3Pipeline); Hugging Face access is gated.",
    ),
    ModelSpec(
        key="sdxl",
        name="SDXL 1.0",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        best_for="Broad style selection, LoRAs, ControlNet, and mature workflows.",
        main_tradeoff="Less reliable text rendering and complex prompt following than newer architectures.",
        license_name="CreativeML Open RAIL++-M License",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        license_notice="The model card lists OpenRAIL++. Review its use restrictions before generation.",
        pipeline_class="StableDiffusionXLPipeline",
        recommended_steps=30,
        hardware=_hardware(
            32, 48, 12, 16,
            model_card="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl",
        ),
        support_status="Supported by Diffusers 0.43.0 (StableDiffusionXLPipeline).",
    ),
    ModelSpec(
        key="pixart-sigma",
        name="PixArt-Σ",
        model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        best_for="Efficient high-resolution generation and experimentation.",
        main_tradeoff="Smaller ecosystem and fewer ready-made workflows.",
        license_name="CreativeML Open RAIL++-M License",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        license_notice="The model card lists OpenRAIL++. Review its use restrictions before generation.",
        pipeline_class="PixArtSigmaPipeline",
        recommended_steps=20,
        hardware=_hardware(
            32, 48, 12, 16,
            model_card="https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pixart_sigma",
        ),
        support_status="Supported by Diffusers 0.43.0 (PixArtSigmaPipeline).",
    ),
    ModelSpec(
        key="sana",
        name="Sana",
        model_id="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        best_for="Efficient, fast high-resolution generation.",
        main_tradeoff="Hardware efficiency is attractive, but tooling and quality vary by version.",
        license_name="Apache License 2.0; Gemma component terms also apply",
        license_url="https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        license_notice="The Sana card lists Apache-2.0 and additional Gemma terms for its text encoder. Review both before use.",
        pipeline_class="SanaPipeline",
        recommended_steps=20,
        hardware=_hardware(
            16, 32, 12, 16,
            model_card="https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            pipeline_docs="https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana",
            upper="The Sana model card describes output up to 4096×4096; no matching upper hardware configuration is verified.",
        ),
        support_status="Supported by Diffusers 0.43.0 (SanaPipeline) using the Diffusers-format model card.",
    ),
)


MODEL_BY_KEY = {model.key: model for model in MODELS}
