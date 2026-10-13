"""Pre-registered content classes and train/dev/confirm splits (P1, docs/P1-plan.md).

Rules, in this order:

* **D** if ``source_type`` is not ``natural`` (graphics, text, screen, animation);
* **C** if ``flat_frac >= 0.5`` or ``dark_frac >= 0.5``;
* **B** if SI or TI is at or above its 66th percentile over the natural clips not in C;
* **A** otherwise.

Splits are 50 / 25 / 25 % per class, grouped by ``source_group`` so that segments of one source
never land in two splits. A source whose segments fall into several classes keeps the split it
got in the first class (A, B, C, D order), so that no source is in the training split of one
class and the confirm split of another.

Clip selectors (``--clips`` / ``--holdout``): ``@A:train``, ``@B:dev``, ``@*:dev``,
``@natural:train``, ``@A`` (every split), mixed freely with plain clip names.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from . import content

CLASSES = ("A", "B", "C", "D")
SPLITS = ("train", "dev", "confirm")
FRACTIONS = {"train": 0.5, "dev": 0.25, "confirm": 0.25}
FLAT_MAX = 0.5
DARK_MAX = 0.5
PERCENTILE = 66
SEED = 0
CLASSES_FILE = Path("docs/p1/classes.json")
NATURAL_CLASSES = ("A", "B", "C")


def assign_classes(clips: dict[str, content.ManifestClip], feats: dict[str, dict]) -> tuple[dict, dict]:
    """-> ({name: class}, thresholds)."""
    missing = sorted(set(clips) - set(feats))
    if missing:
        raise ValueError(f"no features for {missing}")
    cls = {}
    for name, c in clips.items():
        f = feats[name]
        if c.source_type != "natural":
            cls[name] = "D"
        elif f["flat_frac"] >= FLAT_MAX or f["dark_frac"] >= DARK_MAX:
            cls[name] = "C"
    rest = [n for n in clips if n not in cls]
    si_cut = float(np.percentile([feats[n]["si"] for n in rest], PERCENTILE)) if rest else float("nan")
    ti_cut = float(np.percentile([feats[n]["ti"] for n in rest], PERCENTILE)) if rest else float("nan")
    for n in rest:
        cls[n] = "B" if feats[n]["si"] >= si_cut or feats[n]["ti"] >= ti_cut else "A"
    thresholds = {"flat_frac_min_for_C": FLAT_MAX, "dark_frac_min_for_C": DARK_MAX,
                  "percentile_for_B": PERCENTILE, "si_min_for_B": si_cut, "ti_min_for_B": ti_cut,
                  "si_ti": "ITU-T P.910 maximum over frames (ffmpeg siti)"}
    return cls, thresholds


def assign_splits(clips: dict[str, content.ManifestClip], cls: dict[str, str], seed: int = SEED) -> dict[str, str]:
    rng = random.Random(seed)
    group_split: dict[str, str] = {}
    for k in CLASSES:
        members = [n for n in clips if cls[n] == k]
        sizes: dict[str, int] = {}
        for n in members:
            sizes[clips[n].source_group] = sizes.get(clips[n].source_group, 0) + 1
        count = {s: 0 for s in SPLITS}
        for g, size in sizes.items():
            if g in group_split:
                count[group_split[g]] += size
        free = sorted(g for g in sizes if g not in group_split)
        rng.shuffle(free)
        # largest groups first, so the small ones can even out the fractions
        free.sort(key=lambda g: -sizes[g])
        total = len(members)
        for g in free:
            split = max(SPLITS, key=lambda s: (FRACTIONS[s] * total - count[s], -SPLITS.index(s)))
            group_split[g] = split
            count[split] += sizes[g]
    return {n: group_split[clips[n].source_group] for n in clips}


def classify(clips: dict[str, content.ManifestClip], feats: dict[str, dict], seed: int = SEED) -> dict:
    cls, thresholds = assign_classes(clips, feats)
    split = assign_splits(clips, cls, seed)
    counts = {k: {s: sum(1 for n in clips if cls[n] == k and split[n] == s) for s in SPLITS} for k in CLASSES}
    return {"rules": thresholds, "seed": seed, "fractions": FRACTIONS,
            "counts": counts,
            "clips": {n: {"class": cls[n], "split": split[n], "source_group": c.source_group,
                          "source_type": c.source_type} for n, c in clips.items()}}


def write(res: dict, out: Path = CLASSES_FILE) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    return out


def load(path: Path = CLASSES_FILE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run 'vreg classify' first")
    return json.loads(Path(path).read_text())


def select(selector: str, classes: dict) -> list[str]:
    """``@A:train``, ``@*:dev``, ``@natural:train``, ``@B`` -> clip names, in manifest order."""
    what, _, split = selector.lstrip("@").partition(":")
    split = split or "*"
    if what not in (*CLASSES, "*", "natural") or split not in (*SPLITS, "*"):
        raise ValueError(f"bad clip selector {selector!r}; e.g. @A:train, @*:dev, @natural:train")
    ok_cls = NATURAL_CLASSES if what == "natural" else CLASSES if what == "*" else (what,)
    return [n for n, c in classes["clips"].items()
            if c["class"] in ok_cls and (split == "*" or c["split"] == split)]


def resolve(spec: str, classes_file: Path = CLASSES_FILE) -> list[str]:
    """Comma separated clip names and selectors -> names, without duplicates."""
    names, classes = [], None
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if item.startswith("@"):
            classes = classes or load(classes_file)
            names += select(item, classes)
        else:
            names.append(item)
    return list(dict.fromkeys(names))
