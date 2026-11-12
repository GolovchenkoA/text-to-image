import pytest

from text2image.config import ConfigError, ModelCacheConfig, load_model_cache_config


def test_loads_model_cache_settings(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[model_cache]\nenabled = false\nidle_timeout_seconds = 120\n",
        encoding="utf-8",
    )

    assert load_model_cache_config(config_path) == ModelCacheConfig(
        enabled=False,
        idle_timeout_seconds=120,
    )


def test_missing_config_uses_ten_minute_default(tmp_path):
    assert load_model_cache_config(tmp_path / "missing.toml") == ModelCacheConfig()


@pytest.mark.parametrize("value", ["-1", "true", '"600"'])
def test_invalid_idle_timeout_is_rejected(tmp_path, value):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[model_cache]\nidle_timeout_seconds = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="idle_timeout_seconds"):
        load_model_cache_config(config_path)
