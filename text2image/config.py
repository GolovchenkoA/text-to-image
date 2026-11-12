"""Load small, user-editable application settings from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


class ConfigError(ValueError):
    """An invalid application configuration value."""


@dataclass(frozen=True)
class ModelCacheConfig:
    enabled: bool = True
    idle_timeout_seconds: int = 600


def load_model_cache_config(path: Path = DEFAULT_CONFIG_PATH) -> ModelCacheConfig:
    try:
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except FileNotFoundError:
        return ModelCacheConfig()
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    section = data.get("model_cache", {})
    if not isinstance(section, dict):
        raise ConfigError("[model_cache] in config.toml must be a table.")

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("model_cache.enabled must be true or false.")

    timeout = section.get("idle_timeout_seconds", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0:
        raise ConfigError("model_cache.idle_timeout_seconds must be a non-negative integer.")

    return ModelCacheConfig(enabled=enabled, idle_timeout_seconds=timeout)
