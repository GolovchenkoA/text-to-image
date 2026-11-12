"""Command-line interface for model discovery and local generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_model_cache_config
from .model_cache import PipelineCache
from .registry import MODEL_BY_KEY, MODELS, ModelSpec
from .runtime import UserFacingError, check_python_version, generate_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2image",
        description="Generate images locally with Hugging Face Diffusers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list models without downloading weights")

    info_parser = subparsers.add_parser("info", help="show model details")
    info_parser.add_argument("model", choices=tuple(MODEL_BY_KEY))

    generate_parser = subparsers.add_parser("generate", help="generate one image")
    generate_parser.add_argument("--model", required=True, choices=tuple(MODEL_BY_KEY))
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--output", type=Path, default=Path("output.png"))
    generate_parser.add_argument("--width", type=int, default=1024)
    generate_parser.add_argument("--height", type=int, default=1024)
    generate_parser.add_argument("--steps", type=int, default=None)
    generate_parser.add_argument("--seed", type=int, default=None)

    session_parser = subparsers.add_parser(
        "session",
        help="keep one model loaded for multiple interactive generations",
    )
    session_parser.add_argument("--model", required=True, choices=tuple(MODEL_BY_KEY))
    session_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    session_parser.add_argument("--width", type=int, default=1024)
    session_parser.add_argument("--height", type=int, default=1024)
    session_parser.add_argument("--steps", type=int, default=None)
    session_parser.add_argument("--seed", type=int, default=None)
    return parser


def _validate_generation_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        parser.error("--width and --height must be positive multiples of 8")
    model = MODEL_BY_KEY[args.model]
    steps = args.steps if args.steps is not None else model.recommended_steps
    if steps <= 0:
        parser.error("--steps must be a positive integer")
    return steps


def _session(args: argparse.Namespace, steps: int) -> int:
    model = MODEL_BY_KEY[args.model]
    try:
        cache_config = load_model_cache_config()
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cache = PipelineCache(
        idle_timeout_seconds=cache_config.idle_timeout_seconds,
        enabled=cache_config.enabled,
    )
    if cache_config.enabled and cache_config.idle_timeout_seconds > 0:
        print(
            f"Model cache enabled: the loaded pipeline stays in memory for up to "
            f"{cache_config.idle_timeout_seconds} seconds after the last generation."
        )
    else:
        print("Model cache disabled; the pipeline will be released after each generation.")
    print("Enter a prompt to generate an image; type :quit to exit.")

    index = 1
    try:
        while True:
            try:
                prompt = input("Prompt (:quit to exit)> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if prompt.lower() in {":quit", ":q"}:
                break
            if not prompt:
                print("Enter a non-empty prompt.")
                continue

            while (args.output_dir / f"image-{index:03d}.png").exists():
                index += 1
            output_path = args.output_dir / f"image-{index:03d}.png"
            try:
                saved_path = generate_image(
                    model=model,
                    prompt=prompt,
                    output_path=output_path,
                    width=args.width,
                    height=args.height,
                    steps=steps,
                    seed=args.seed,
                    cache=cache,
                )
            except UserFacingError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                continue
            except ImportError:
                print(
                    "ERROR: A model runtime dependency is missing. Install "
                    "requirements.txt in the project .venv.",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:
                print(f"ERROR: Generation failed: {exc}", file=sys.stderr)
                continue
            print(f"Image saved to: {saved_path}")
            index += 1
    finally:
        cache.close()
    return 0


def _print_model_list() -> None:
    for model in MODELS:
        print(f"{model.key} — {model.name}")
        print(f"  Best for: {model.best_for}")
        print(f"  Main trade-off: {model.main_tradeoff}")
        print(f"  Support: {model.support_status}")


def _print_model_info(model: ModelSpec) -> None:
    hardware = model.hardware
    print(f"{model.name} ({model.key})")
    print(f"Model ID: {model.model_id}")
    print(f"Best for: {model.best_for}")
    print(f"Main trade-off: {model.main_tradeoff}")
    print(f"Support status: {model.support_status}")
    print(f"Pipeline: {model.pipeline_class}")
    print(f"License: {model.license_name}")
    print(f"License notice: {model.license_notice}")
    print(f"License source: {model.license_url}")
    print("Hardware estimates (planning guidance, not hard limits):")
    print(
        f"  System RAM: {hardware.system_ram_min_gb} GB minimum estimate; "
        f"{hardware.system_ram_recommended_gb} GB recommended"
    )
    print(
        f"  GPU VRAM: {hardware.gpu_vram_min_gb} GB minimum estimate; "
        f"{hardware.gpu_vram_recommended_gb} GB recommended"
    )
    print(f"  Practical upper configuration: {hardware.practical_upper}")
    print(f"  CPU: {hardware.cpu_note}")
    print(f"  Assumptions: {hardware.assumptions}")
    print("  Estimate sources:")
    for source in hardware.sources:
        print(f"    {source}")
    print(f"Recommended default inference steps: {model.recommended_steps}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    try:
        check_python_version()
    except UserFacingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        _print_model_list()
        return 0
    if args.command == "info":
        _print_model_info(MODEL_BY_KEY[args.model])
        return 0
    if args.command == "generate":
        model = MODEL_BY_KEY[args.model]
        steps = _validate_generation_options(parser, args)
        if not args.prompt.strip():
            parser.error("--prompt must not be empty")
        try:
            saved_path = generate_image(
                model=model,
                prompt=args.prompt,
                output_path=args.output,
                width=args.width,
                height=args.height,
                steps=steps,
                seed=args.seed,
            )
        except UserFacingError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except ImportError as exc:
            print(
                "ERROR: A model runtime dependency is missing. Install "
                "requirements.txt in the project .venv.",
                file=sys.stderr,
            )
            return 2
        except Exception as exc:
            print(f"ERROR: Generation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Image saved to: {saved_path}")
        return 0
    if args.command == "session":
        steps = _validate_generation_options(parser, args)
        return _session(args, steps)
    return 2
