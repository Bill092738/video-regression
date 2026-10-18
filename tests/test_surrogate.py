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


def test_variant_params_depend_on_clip_and_index_only():
    from vreg import precomp
    from vreg.surrogate.loop import variant_params

    assert variant_params("a", 0) == precomp.normalize({})
    assert variant_params("a", 3) == variant_params("a", 3)
    assert variant_params("a", 3) != variant_params("b", 3) != variant_params("a", 4)
    assert all(variant_params("a", i)["upscale"] == 1.0 for i in range(6))


def test_pairs_cache_is_reused_without_the_platform(tiny_clip, tmp_path):
    from test_pipeline import TINY

    from vreg.platform import SimulatedPlatform
    from vreg.surrogate.loop import collect_random_pairs, pairs_dir

    plat = SimulatedPlatform(TINY)
    p1 = collect_random_pairs({"tiny": tiny_clip}, plat, 2.0, 2, tmp_path, log=lambda *_: None)
    d = pairs_dir(tmp_path, plat, 2.0)
    assert sorted(f.name for f in (d / "tiny_1").iterdir()) == ["params.json", "v.npy", "y.npy"]  # no videos kept

    def boom(*_a, **_k):
        raise AssertionError("platform called on a cache hit")

    plat.process = boom
    p2 = collect_random_pairs({"tiny": tiny_clip}, plat, 2.0, 2, tmp_path, log=lambda *_: None)
    assert len(p2.inputs) == 2 and np.array_equal(p1.outputs[1], p2.outputs[1])


@pytest.mark.parametrize("g_loss", ["lpips", "dists", "lpips+dists"])
def test_perceptual_losses_are_differentiable(g_loss):
    pytest.importorskip("piq")
    from vreg.surrogate.loop import perceptual_loss

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(1, 3, 64, 64, device=dev)
    y = (x + 0.1 * torch.rand_like(x)).clamp(0, 1).requires_grad_(True)
    loss = perceptual_loss(g_loss)(y, x)
    loss.backward()
    assert loss.item() > 0 and y.grad.abs().sum() > 0


def test_anchored_forward_equals_real_output_at_the_anchor():
    """E7: at the anchor, E(G(X)) + delta is the black box's output; the gradient is E's."""
    import random

    from vreg.surrogate.loop import train_generator_anchored

    torch.manual_seed(0)
    src = [np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8)]
    E, G = emulator(width=8), generator(width=8, max_residual=0.06)
    seen = []

    def anchor_fn(g):
        out = apply_full_frames(g, src[0], "cpu")
        y_sim = apply_full_frames(E, out, "cpu")
        y_real = np.clip(out.astype(int) + 7, 0, 255)  # a "black box" E does not know: +7 levels
        seen.append(1)
        return [(y_real - y_sim).astype(np.int16)]

    def perceptual(y_hat, x):
        return (y_hat - x).abs().mean()

    losses, log = train_generator_anchored(G, E, src, 6, 3, anchor_fn, "cpu", random.Random(0), perceptual,
                                           lr=1e-3, l1_weight=0.0, log=lambda *_: None)
    assert len(seen) == 3 and len(losses) == 6
    assert log[0]["step"] == 0 and log[1]["step"] == 2
    assert 6.0 < log[0]["mean_abs_offset"] < 8.0  # the offset measures the emulator's error


def test_theta_front_end_channel(tiny_clip, tmp_path):
    """E9: a theta in front of the black box; its pairs are cached apart from the upscale-only ones."""
    from test_pipeline import TINY

    from vreg import ff, precomp
    from vreg.platform import SimulatedPlatform
    from vreg.surrogate.loop import channel, pairs_dir, send_through_platform

    theta = {"upscale": 2.0, "sharpen": 1.0, "sharpen_radius": 2.0}
    assert channel(2.0) == precomp.normalize({"upscale": 2.0})
    assert channel(theta)["sharpen"] == 1.0
    plat = SimulatedPlatform(TINY)
    assert pairs_dir(tmp_path, plat, 2.0) != pairs_dir(tmp_path, plat, channel(theta))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = send_through_platform(tiny_clip, 2.0, plat, tmp_path / "a")
    b = send_through_platform(tiny_clip, channel(theta), plat, tmp_path / "b")
    fa, fb = ff.read_frames(a), ff.read_frames(b)
    assert fa.shape == fb.shape and not np.array_equal(fa, fb)
