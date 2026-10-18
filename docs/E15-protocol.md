# E15/E16 protocol: the minimal online-platform validation

**Goal.** Decide, for one online platform, whether the recipe that the simulator studies support
should be used there:
* upload at 1440p (×2 lanczos), near-lossless;
* add a mild unsharp (luma 5×5, amount 0.5) on natural low-motion content (class A);
* use tier-only on everything else.

The budget is as few uploads as possible: **23 per platform**. It follows from E10 (validation
size), E11 (recipe, upload bitrate) and E12 (passthrough probe); see PROGRESS §11–§13.

This is a plan only. Nothing in the repository uploads anything by itself; `CommandPlatform`
runs only the commands you give it, with your own account.

## Rules that hold throughout

* Use your own account only, within the platform's terms of service. Set every upload to
  **private** (YouTube "Private"; Bilibili "仅自己可见"). Delete the uploads when the measurements
  are done.
* **Upload only material you may share with a third-party service.** The clips below come from
  sources with no copyright and no restrictions (Taurus Media Technik/TUM, SVT via TUM) or listed
  as public domain (the video-conference sequences); see `docs/CREDITS.md`. Never upload the
  research-only sources: NTIA/ITS, SVT MultiFormat, HDgreetings. Your own recordings are fine.
* Leave time between uploads, at least a few minutes (`CommandPlatform` enforces a minimum gap), and
  stop at the budget (`CommandPlatform` enforces a maximum count).
* Download only after the platform has finished processing. Some platforms add better codecs
  (VP9, AV1) hours after the first rendition, so download each upload **at least 24 h after
  upload**, and record the served formats (`yt-dlp -F`) at download time.

## Step 1: validation (20 uploads)

10 class A clips, each uploaded twice: tier-only (`{"upscale": 2}`, CRF 10) and the mild recipe
(`{"upscale": 2, "sharpen": 0.5, "sharpen_radius": 2}`, CRF 10). The selection follows E10's
"hard-half" rule: five clips from class A's top third by SI, and five more chosen for variety of
source and scene. All ten are free to upload.

| # | clip | SI | TI | why |
|---|---|---|---|---|
| 1 | mobcal_6 | 109.1 | 32.5 | top-third SI |
| 2 | stockholm_6 | 96.5 | 27.6 | top-third SI |
| 3 | kristen_and_sara_2 | 87.4 | 14.9 | top-third SI |
| 4 | stockholm_0 | 85.5 | 28.2 | top-third SI |
| 5 | four_people_2 | 80.1 | 12.2 | top-third SI |
| 6 | vidyo3_2 | 75.0 | 15.7 | video conference |
| 7 | tractor_0 | 69.0 | 25.9 | camera pan, agriculture |
| 8 | sunflower_10 | 62.2 | 29.2 | close-up, fine detail |
| 9 | pedestrian_area_0 | 54.4 | 24.4 | street, people close to the camera |
| 10 | station2_8 | 48.1 | 7.3 | slow zoom, regular structures |

**Decision, fixed now:** switch the recipe on for class A on this platform iff, over the 10 clips,
its mean gain over tier-only is ≥ 0 on **both** SSIMULACRA2 and XPSNR. Otherwise keep tier-only
everywhere.
* A wrong "off" costs at most the small simulated gain (about +0.4 SSIMULACRA2).
* A wrong "on" can cost a clip much more (−25 on a rate-starved AV1 clip, §8). So when in doubt,
  keep tier-only.

DISTS is reported, but not used in the decision.

## Step 2: passthrough probe (2 uploads)

johnny_2 (low motion, TI 9.7) at 720p, no upscale, H.264 at **1.9 Mbps** and at **2.5 Mbps**.
* The platform has a passthrough rule for small uploads if the 1.9 Mbps upload comes back
  bit-identical, or at its upload bitrate with the same codec, while the 2.5 Mbps one is
  re-encoded.
* E12 found such a rule not worth exploiting in the simulator. If a real platform has one, its cap
  and the quality on both sides decide whether that holds there.

## Step 3: upload bitrate (1 upload)

mobcal_6 at ×2 uploaded at **16 Mbps** (YouTube's recommended SDR 1440p rate). Its tier-only
upload from step 1 serves as the near-lossless pair. E11 predicts a clear loss at 16 Mbps (about
−1.9 SSIMULACRA2 in the simulator). One clip is a sanity check, not a test.

## What to record (one row per upload)

`platform, clip, config, upload_file, upload_kbps, upload_time, platform_id, download_time,
served_formats (yt-dlp -F), downloaded_format, playback_file`

The metrics are computed locally, exactly as in the simulator (`vreg rescore`-style: playback vs
source, with VMAF NEG, LPIPS, SSIMULACRA2, DISTS and XPSNR).

## Platform notes (check before starting; these change)

* **YouTube.**
  * Uploads through the official YouTube Data API need an OAuth client; the API's daily quota
    allows only a handful of uploads a day, so 23 uploads may take several days. Check the current
    quota in the Google Cloud console.
  * The web upload page has no such limit but is manual.
  * Downloads: `yt-dlp -F <id>` lists the renditions. Download the 720p video-only stream for
    comparability with the simulator, plus the highest one served.
* **Bilibili.**
  * Uploads through the web page or an upload CLI (for example `biliup`).
  * Uploads go through review before they become playable.
  * Higher renditions may require a logged-in session for downloading (`yt-dlp --cookies`).

## Budget

| step | uploads |
|---|---|
| 1 validation | 20 |
| 2 passthrough probe | 2 |
| 3 upload bitrate | 1 |
| **total per platform** | **23** |

Optional, only if step 1 is inconclusive (mean gains within ±0.1 of 0): add 4 more class A clips
from the free list (+8 uploads), then decide with the same rule on all 14.
