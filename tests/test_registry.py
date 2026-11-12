from text2image.registry import MODELS


def test_registry_has_all_requested_models_and_metadata():
    assert len(MODELS) == 7
    assert len({model.key for model in MODELS}) == 7
    for model in MODELS:
        assert model.model_id
        assert model.best_for
        assert model.main_tradeoff
        assert model.license_name
        assert model.license_url.startswith("https://")
        assert model.pipeline_class
        assert model.support_status.startswith("Supported")
        assert model.hardware.system_ram_min_gb > 0
        assert model.hardware.gpu_vram_min_gb > 0
        assert model.hardware.sources


def test_required_tradeoff_text_is_present():
    by_name = {model.name: model for model in MODELS}
    assert "noncommercial restrictions" in by_name["FLUX.1 [dev]"].main_tradeoff
    assert "very few steps" in by_name["FLUX.1 [schnell]"].main_tradeoff
    assert "Stability’s license" in by_name["Stable Diffusion 3.5 Large"].main_tradeoff

