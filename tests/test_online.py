"""vreg online: phases, ledger, hard upload budget, minimum gap, decision (with a fake local 'platform')."""

import json

import pytest

from vreg import content, online

RECIPE = {"upscale": 2.0, "sharpen": 0.5, "sharpen_radius": 2.0}


@pytest.fixture
def run(tiny_clip, tmp_path, monkeypatch):
    monkeypatch.setattr(content, "ensure_clips", lambda names, _d: {n: tiny_clip for n in names})
    proto = {"max_uploads": 4, "min_gap_s": 60,
             "decision": {"step": "validation", "on": "recipe", "off": "tier", "judges": ["ssimulacra2", "xpsnr"]},
             "items": [{"step": "validation", "clip": c, "config": cfg, "params": p}
                       for c in ("a", "b") for cfg, p in (("tier", {"upscale": 2.0}), ("recipe", RECIPE))]}
    f = tmp_path / "proto.json"
    f.write_text(json.dumps(proto))
    store = tmp_path / "store"
    store.mkdir()
    up = f"cp {{file}} {store}/ && basename {{file}}"
    down = f"cp {store}/{{id}} {{out}}"
    return tmp_path / "run", f, up, down


def test_phases_and_resume(run):
    run_dir, proto, up, down = run
    online.prepare(proto, run_dir, log=lambda *_: None)
    waits = []
    online.upload(run_dir, up, sleep=waits.append, log=lambda *_: None)
    led = online.Ledger(run_dir / "ledger.jsonl")
    assert len(led.latest("uploaded")) == 4
    assert len(waits) == 3 and all(0 < w <= 60 for w in waits)  # a minimum gap between consecutive uploads
    online.upload(run_dir, up, sleep=waits.append, log=lambda *_: None)  # resuming uploads nothing twice
    assert len([e for e in led.entries() if e["event"] == "uploaded"]) == 4
    online.download(run_dir, down, log=lambda *_: None)
    assert len(led.latest("downloaded")) == 4


def test_budget_is_a_hard_cap(run):
    run_dir, proto, up, _ = run
    p = json.loads(proto.read_text())
    p["items"].append({"step": "extra", "clip": "c", "config": "tier", "params": {"upscale": 2.0}})
    proto.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="max_uploads"):
        online.prepare(proto, run_dir, log=lambda *_: None)  # a protocol over budget is refused up front
    p["items"].pop()
    proto.write_text(json.dumps(p))
    online.prepare(proto, run_dir, log=lambda *_: None)
    pr = json.loads((run_dir / "protocol.json").read_text())
    pr["max_uploads"] = 2  # e.g. a smaller budget set after preparing
    (run_dir / "protocol.json").write_text(json.dumps(pr))
    with pytest.raises(online.BudgetExceeded):
        online.upload(run_dir, up, sleep=lambda *_: None, log=lambda *_: None)
    assert len(online.Ledger(run_dir / "ledger.jsonl").latest("uploaded")) == 2


def test_decision_rule():
    proto = {"decision": {"step": "validation", "on": "recipe", "off": "tier", "judges": ["ssimulacra2", "xpsnr"]},
             "items": [{"step": "validation", "clip": c, "config": cfg} for c in ("a", "b") for cfg in ("tier", "recipe")]}
    rows = {"a__tier": {"ssimulacra2": 50, "xpsnr": 30}, "a__recipe": {"ssimulacra2": 51, "xpsnr": 30.5},
            "b__tier": {"ssimulacra2": 60, "xpsnr": 31}, "b__recipe": {"ssimulacra2": 59.5, "xpsnr": 31.2}}
    assert online.decide(rows, proto)["decision"] == "on"
    rows["b__recipe"]["ssimulacra2"] = 58
    assert online.decide(rows, proto)["decision"] == "off"  # mean SSIMULACRA2 gain < 0
    del rows["b__recipe"]
    assert online.decide(rows, proto)["decision"] == "incomplete"
