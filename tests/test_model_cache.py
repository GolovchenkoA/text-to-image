from text2image.model_cache import PipelineCache


def test_reuses_same_model_without_loading_twice():
    cache = PipelineCache(idle_timeout_seconds=600)
    load_calls = []
    pipeline = object()
    try:
        first, first_was_cached = cache.get_or_load(
            "model:cpu:float32", lambda: load_calls.append("load") or pipeline
        )
        cache.mark_idle()
        second, second_was_cached = cache.get_or_load(
            "model:cpu:float32", lambda: load_calls.append("load") or object()
        )

        assert first is pipeline
        assert second is pipeline
        assert first_was_cached is False
        assert second_was_cached is True
        assert load_calls == ["load"]
    finally:
        cache.close()


def test_zero_timeout_evicts_after_generation():
    cache = PipelineCache(idle_timeout_seconds=0)
    try:
        pipeline, _ = cache.get_or_load("model", object)
        cache.mark_idle()

        assert cache.get_or_load("model", object)[0] is not pipeline
    finally:
        cache.close()
