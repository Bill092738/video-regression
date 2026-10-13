"""P1 additions: clip manifest, content features, classes and splits, clip selectors."""

import csv
import json

import numpy as np
import pytest

from vreg import classify, content, features, ff
from vreg.content import ManifestClip

ROW = {"source_group": "g", "start_s": "0", "source_type": "synthetic", "license": "own", "license_url": "-"}


def write_manifest(path, rows):
    cols = [*content.MANIFEST_COLUMNS, "src_fps"]
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in cols})
    return path


def mclip(name, group="g", source_type="natural", url="x.mkv", start=0.0, fps=None):
    return ManifestClip(name, group, url, start, source_type, "own", "-", fps)


def test_manifest_rejects_missing_license_and_bad_type(tmp_path):
    p = write_manifest(tmp_path / "m.csv", [{**ROW, "name": "a", "url_or_path": "lavfi:x", "license": ""}])
    with pytest.raises(ValueError, match="lacks"):
        content.read_manifest(p)
    p = write_manifest(tmp_path / "m.csv", [{**ROW, "name": "a", "url_or_path": "lavfi:x", "source_type": "tv"}])
    with pytest.raises(ValueError, match="source_type"):
        content.read_manifest(p)


def test_manifest_keeps_p0_clip_only_with_its_recipe(tmp_path):
    ok = {**ROW, "name": "park_joy", "url_or_path": content.XIPH_BASE + content.XIPH_CLIPS["park_joy"][0]}
    assert "park_joy" in content.read_manifest(write_manifest(tmp_path / "m.csv", [ok]))
    bad = {**ok, "start_s": "5"}
    with pytest.raises(ValueError, match="P0 clip"):
        content.read_manifest(write_manifest(tmp_path / "m.csv", [bad]))


def test_fetch_segment_offset_and_retime(tmp_path):
    # a 25 fps source whose frame number is burnt into the luma level
    src = tmp_path / "src.mkv"
    ff.run(["-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=8", "-vf", "geq=lum='2*N':cb=128:cr=128",
            *content.MASTER_CODEC, str(src)])
    out = content.fetch_segment(mclip("s", url=str(src), start=2.0), tmp_path / "s.mkv")
    info = ff.probe(out)
    assert (info.width, info.height, info.frames) == (1280, 720, 90) and info.fps == pytest.approx(30)
    y = features.read_luma(out, step=1)[:, 360, 640].astype(int)
    assert abs(y[0] - 100) <= 3  # starts at source frame 50 = 2 s, frame for frame
    assert np.all(np.diff(y[:40]) > 0)  # no duplicated frames


def test_fetch_segment_too_short_fails(tmp_path):
    src = tmp_path / "short.mkv"
    ff.run(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=2", *content.MASTER_CODEC, str(src)])
    with pytest.raises(ff.FFmpegError, match="too short"):
        content.fetch_segment(mclip("s", url=str(src)), tmp_path / "s.mkv")
    assert not (tmp_path / "s.mkv").exists()


def test_fetch_frames_pattern_crops_to_16_9(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    ff.run(["-f", "lavfi", "-i", "testsrc2=s=470x200:r=24:d=5", "-start_number", "1", str(frames / "%04d.png")])
    out = content.fetch_segment(mclip("p", url=str(frames / "%04d.png"), start=1.0, fps=24, source_type="animation"),
                                tmp_path / "p.mkv")
    info = ff.probe(out)
    assert (info.width, info.height, info.frames) == (1280, 720, 90)


def test_features_flat_and_dark(tmp_path):
    flat = tmp_path / "flat.mkv"
    ff.run(["-f", "lavfi", "-i", "color=c=0x101010:s=320x192:r=30:d=1", *content.MASTER_CODEC, str(flat)])
    f = features.clip_features("flat", flat)
    assert f["flat_frac"] == 1.0 and f["dark_frac"] == 1.0 and f["si"] < 1 and f["ti"] < 1


def test_features_busy_clip(tiny_clip):
    f = features.clip_features("tiny", tiny_clip)
    assert f["si"] > 10 and f["flat_frac"] < 0.9 and 0 <= f["dark_frac"] < 0.5
    assert f["si"] >= f["si_mean"] and f["ti"] >= f["ti_mean"]


def test_features_csv_roundtrip(tmp_path):
    row = {"name": "a", "si": 1.0, "ti": 2.0, "si_mean": 0.5, "ti_mean": 1.0, "flat_frac": 0.1,
           "dark_frac": 0.2, "mean_luma": 90.0}
    assert features.read_csv(features.write_csv([row], tmp_path / "f.csv"))["a"] == row


def feats(si, ti, flat=0.0, dark=0.0):
    return {"si": si, "ti": ti, "flat_frac": flat, "dark_frac": dark}


def test_class_rules_in_order():
    clips = {"anim": mclip("anim", source_type="animation"), "dark": mclip("dark"), "flat": mclip("flat"),
             **{f"n{i}": mclip(f"n{i}") for i in range(6)}}
    fs = {"anim": feats(99, 99, dark=0.9), "dark": feats(99, 99, dark=0.6), "flat": feats(1, 1, flat=0.5),
          **{f"n{i}": feats(10 * i, i) for i in range(5)}, "n5": feats(1, 50)}
    cls, thr = classify.assign_classes(clips, fs)
    assert cls["anim"] == "D" and cls["dark"] == "C" and cls["flat"] == "C"
    # percentiles over the six natural non-C clips only
    assert thr["si_min_for_B"] == pytest.approx(np.percentile([0, 10, 20, 30, 40, 1], 66))
    assert cls["n5"] == "B" and cls["n4"] == "B" and cls["n0"] == "A" and cls["n1"] == "A"


def test_splits_grouped_by_source_and_shared_across_classes():
    clips = {f"c{i}": mclip(f"c{i}", group=f"g{i // 2}") for i in range(24)}
    cls = {n: "A" if i < 16 else "B" for i, n in enumerate(clips)}
    cls["c15"] = "B"  # g7 is split across classes A and B
    split = classify.assign_splits(clips, cls, seed=0)
    for i in range(0, 24, 2):
        assert split[f"c{i}"] == split[f"c{i + 1}"]
    a = [split[n] for n in clips if cls[n] == "A"]
    assert a.count("train") >= 6 and a.count("dev") >= 3 and a.count("confirm") >= 3
    assert split == classify.assign_splits(clips, cls, seed=0)


def test_selectors(tmp_path):
    res = {"clips": {"a1": {"class": "A", "split": "train"}, "a2": {"class": "A", "split": "dev"},
                     "b1": {"class": "B", "split": "dev"}, "d1": {"class": "D", "split": "dev"}}}
    f = tmp_path / "classes.json"
    f.write_text(json.dumps(res))
    assert classify.resolve("@A:train", f) == ["a1"]
    assert classify.resolve("@*:dev", f) == ["a2", "b1", "d1"]
    assert classify.resolve("@natural:dev", f) == ["a2", "b1"]
    assert classify.resolve("@A,b1,a1", f) == ["a1", "a2", "b1"]
    assert classify.resolve("park_joy,old_town_cross", f) == ["park_joy", "old_town_cross"]
    with pytest.raises(ValueError):
        classify.resolve("@E:train", f)


def test_g_is_not_judged_by_its_own_loss():
    from vreg.rescore import judges_for

    assert judges_for("lpips") == ("ssimulacra2", "dists", "xpsnr")
    assert judges_for(None) == ("ssimulacra2", "dists", "xpsnr")
    for loss in ("dists", "lpips+dists"):
        assert "dists" not in judges_for(loss) and "vmaf_neg" in judges_for(loss)
