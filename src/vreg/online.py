"""Minimal online-platform validation (docs/E15-protocol.md), in separate phases, with a hard upload budget.

    vreg online prepare  --protocol docs/e15_protocol.json --dir runs/online/youtube
    vreg online upload   --dir runs/online/youtube --upload-cmd '<your upload command with {file}>'
    vreg online download --dir runs/online/youtube --download-cmd '<your download command with {id} {out}>'
    vreg online score    --dir runs/online/youtube

* ``prepare`` renders every upload file locally (nothing leaves the machine).
* ``upload`` runs your upload command once per file, never more than the protocol's
  ``max_uploads`` in total (counted from the ledger, across runs), and at least ``min_gap_s``
  apart. The command must print the platform's video id or URL on its last line of stdout.
* ``download`` runs your download command for every uploaded file that has no playback yet. Run it
  at least 24 h after uploading (the protocol), so that later renditions are available.
* ``score`` computes the metrics (playback vs source, as in the simulator) and the protocol's
  decision.

Every action is appended to ``ledger.jsonl`` in the run directory, which is the record of what was
uploaded. Use only your own account and material you may upload, within the platform's terms.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from . import content, ff, metrics, precomp


class BudgetExceeded(RuntimeError):
    pass


def item_key(item: dict) -> str:
    return f"{item['clip']}__{item['config']}"


class Ledger:
    """Append-only JSONL record of every prepared, uploaded and downloaded file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def add(self, **rec) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), **rec}) + "\n")

    def latest(self, event: str) -> dict[str, dict]:
        out = {}
        for e in self.entries():
            if e.get("event") == event:
                out[e["key"]] = e
        return out


def load_protocol(path: Path) -> dict:
    proto = json.loads(Path(path).read_text())
    keys = [item_key(i) for i in proto["items"]]
    if len(set(keys)) != len(keys):
        raise ValueError("protocol items must have unique (clip, config)")
    if len(keys) > proto["max_uploads"]:
        raise ValueError(f"protocol has {len(keys)} items but max_uploads is {proto['max_uploads']}")
    return proto


def prepare(proto_path: Path, run_dir: Path, log=print) -> None:
    proto = load_protocol(proto_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(json.dumps(proto, indent=2))
    ledger = Ledger(run_dir / "ledger.jsonl")
    done = ledger.latest("prepared")
    clips = content.ensure_clips(sorted({i["clip"] for i in proto["items"]}), content.CLIPS_DIR)
    for item in proto["items"]:
        key = item_key(item)
        out = run_dir / "uploads" / f"{key}.mp4"
        if key in done and out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        precomp.apply(clips[item["clip"]], item["params"], out, upload_kbps=item.get("upload_kbps"))
        info = ff.probe(out)
        ledger.add(event="prepared", key=key, file=str(out), upload_kbps=info.bitrate / 1000, height=info.height)
        log(f"prepared {key}: {info.height}p, {info.bitrate / 1000:.0f} kbps")


def upload(run_dir: Path, upload_cmd: str, sleep=time.sleep, log=print) -> None:
    proto = json.loads((run_dir / "protocol.json").read_text())
    ledger = Ledger(run_dir / "ledger.jsonl")
    prepared = ledger.latest("prepared")
    for item in proto["items"]:
        key = item_key(item)
        uploaded = ledger.latest("uploaded")
        if key in uploaded:
            continue
        if len(uploaded) >= proto["max_uploads"]:
            raise BudgetExceeded(f"upload budget of {proto['max_uploads']} reached; nothing more is uploaded")
        if key not in prepared:
            raise RuntimeError(f"{key} is not prepared; run 'vreg online prepare' first")
        if uploaded:
            wait = proto["min_gap_s"] - (time.time() - max(e["time"] for e in uploaded.values()))
            if wait > 0:
                log(f"waiting {wait:.0f} s before the next upload")
                sleep(wait)
        file = prepared[key]["file"]
        proc = subprocess.run(upload_cmd.format(file=shlex.quote(file)), shell=True, capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            ledger.add(event="upload_failed", key=key, stderr=proc.stderr[-500:])
            raise RuntimeError(f"upload of {key} failed: {proc.stderr[-500:]}")
        vid = proc.stdout.strip().splitlines()[-1]
        ledger.add(event="uploaded", key=key, id=vid, file=file)
        log(f"uploaded {key} -> {vid} ({len(uploaded) + 1}/{proto['max_uploads']})")


def download(run_dir: Path, download_cmd: str, log=print) -> None:
    ledger = Ledger(run_dir / "ledger.jsonl")
    have = ledger.latest("downloaded")
    for key, e in ledger.latest("uploaded").items():
        if key in have and Path(have[key]["file"]).exists():
            continue
        out = run_dir / "playback" / f"{key}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = download_cmd.format(id=shlex.quote(e["id"]), out=shlex.quote(str(out)))
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            ledger.add(event="download_failed", key=key, stderr=proc.stderr[-500:])
            log(f"download of {key} not available yet: {proc.stderr[-200:].strip()}")
            continue
        info = ff.probe(out)
        ledger.add(event="downloaded", key=key, file=str(out), codec=info.codec, height=info.height,
                   playback_kbps=info.bitrate / 1000)
        log(f"downloaded {key}: {info.codec} {info.height}p {info.bitrate / 1000:.0f} kbps")


def decide(rows: dict[str, dict], proto: dict) -> dict:
    """The protocol's rule: 'on' iff the mean gain of `on` over `off` is >= 0 on every listed judge."""
    d = proto["decision"]
    items = [i for i in proto["items"] if i["step"] == d["step"]]
    clips = sorted({i["clip"] for i in items})
    gains = {}
    for j in d["judges"]:
        g = [rows[f"{c}__{d['on']}"][j] - rows[f"{c}__{d['off']}"][j] for c in clips
             if f"{c}__{d['on']}" in rows and f"{c}__{d['off']}" in rows]
        gains[j] = (float(np.mean(g)) if g else float("nan"), len(g))
    complete = all(n == len(clips) for _, n in gains.values())
    on = complete and all(m >= 0 for m, _ in gains.values())
    return {"gains": gains, "complete": complete, "decision": ("on" if on else "off") if complete else "incomplete"}


def score(run_dir: Path, log=print) -> dict:
    proto = json.loads((run_dir / "protocol.json").read_text())
    ledger = Ledger(run_dir / "ledger.jsonl")
    clips = content.ensure_clips(sorted({i["clip"] for i in proto["items"]}), content.CLIPS_DIR)
    by_key = {item_key(i): i for i in proto["items"]}
    rows = {}
    for key, e in ledger.latest("downloaded").items():
        m = metrics.compute(Path(e["file"]), clips[by_key[key]["clip"]], judges=metrics.JUDGE_METRICS)
        rows[key] = m
        log(f"{key}: SSIMULACRA2 {m.get('ssimulacra2', float('nan')):.2f}  XPSNR {m.get('xpsnr', float('nan')):.2f}")
    res = {"rows": rows, "decision": decide(rows, proto)}
    (run_dir / "score.json").write_text(json.dumps(res, indent=2))
    log(f"decision: {res['decision']['decision']} {res['decision']['gains']}")
    return res
