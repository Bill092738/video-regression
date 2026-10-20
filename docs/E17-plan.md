# E17 plan: a small paired-comparison viewer study

**Why.** Every result so far rests on objective metrics, and several of them disagree:
* the learned `G` wins on LPIPS and loses on SSIMULACRA2 (P1, E7, E9);
* the filter gains are small (E11).

A small two-alternative forced-choice (2AFC) study with 5–10 viewers is the cheapest way to see
which side people agree with. It needs no online platform: the stimuli are the simulator's
playbacks.

## Comparisons (each against tier-only, ×2 CRF 10, on `sim:ugc_strict`)

| id | treatment | question it answers |
|---|---|---|
| `recipe` | ×2 + mild unsharp (5×5, 0.5), the gate's choice for class A | Do people prefer the recipe the metrics support? |
| `learned_g` | the P1 global `G` (loss LPIPS, seed 0) | When LPIPS and SSIMULACRA2 disagree, which side do people take? |
| `bitrate` | ×2 uploaded at 16 Mbps (the reference is near-lossless) | **Positive control:** a clear predicted loss; viewers who cannot see it are flagged |

## Stimuli

* The 10 class A dev and confirm clips that are free to share: vidyo1_2, vidyo3_2, vidyo4_2,
  four_people_2, johnny_2, kristen_and_sara_2, pedestrian_area_0, pedestrian_area_8, tractor_0 and
  tractor_12. The SVT and NTIA dev clips are left out.
* Each stimulus is the simulator's 720p playback of that configuration. It is re-encoded once with
  x264 CRF 4 (High profile, 4:2:0) so that every browser can play it; both sides of a pair get the
  same re-encode.

## Procedure

* 30 trials (10 clips × 3 comparisons) in random order, plus 3 repeated trials as a consistency
  check, so about 10 minutes per viewer.
* Each trial shows the two videos as **A and B in the same place at 100 % size** (1280×720). The
  viewer switches between them with keys 1 and 2 (or the buttons) while they loop in sync. Two
  720p videos side by side would not fit most screens at 100 %. Which treatment is A is randomised
  per trial.
* The viewer must choose the one that looks better (no "same" button) and can replay freely.
* One machine and display for everyone (a local HTML page; `runs/e17/index.html`), in normal room
  light, at a viewing distance of about 3 picture heights.
* Results are saved in the browser and downloaded as one JSON file per viewer.

## Analysis (fixed now; `scripts/e17_analyze.py`)

* For each comparison, the preference rate for the treatment, with a 95 % CI from a cluster
  bootstrap over viewers and clips (10,000 resamples).
* **Success criterion** (backlog E17): the treatment is preferred if the rate is above 50 % and its
  CI excludes 50 %; dispreferred if below with the CI excluding 50 %; otherwise no preference.
* **Exclusions:** a viewer is excluded if they prefer 16 Mbps over near-lossless in more than half
  of the `bitrate` trials, or disagree with themselves in 2 or more of the 3 repeated trials.
  Exclusions are reported.

This is a small study; it can show a clear preference, not a small one. It changes no metric-based
verdict, but the paper reports it next to them.
