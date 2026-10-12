import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vreg.surrogate.loop import PairSet  # noqa: E402
from vreg.surrogate.models import apply_full_frames, emulator, generator  # noqa: E402

pytestmark = pytest.mark.gpu


def test_nets_start_as_identity():
    x = torch.rand(2, 3, 64, 64)
    for net in (emulator(width=8), generator(width=8)):
        assert torch.allclose(net(x), x)


def test_generator_residual_is_bounded():
    g = generator(width=8, max_residual=0.1)
    torch.nn.init.normal_(g.out.weight, std=10.0)
    x = torch.rand(1, 3, 64, 64)
    assert (g(x) - x).abs().max() <= 0.1 + 1e-6


def test_apply_full_frames_pads_odd_sizes():
    frames = np.random.randint(0, 255, (3, 50, 70, 3), dtype=np.uint8)
    out = apply_full_frames(generator(width=8), frames, "cpu")
    assert out.shape == frames.shape
    assert np.abs(out.astype(int) - frames).max() <= 1  # identity up to rounding


def test_pairset_crops_are_aligned():
    import random

    v = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
    ps = PairSet()
    ps.add(v, v.copy(), "same")
    a, b = ps.sample(4, random.Random(0), "cpu")
    assert a.shape == (4, 3, 256, 256) and torch.equal(a, b)
