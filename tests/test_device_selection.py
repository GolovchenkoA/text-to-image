from types import SimpleNamespace

from text2image.runtime import select_device


def test_selects_cuda_when_available():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    assert select_device(torch) == "cuda"


def test_falls_back_to_cpu_when_cuda_unavailable():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert select_device(torch) == "cpu"

