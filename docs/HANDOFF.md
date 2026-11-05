# Vidbush handoff

Vidbush turns a brief, a script, or a finished voiceover into a publish-ready
long-form documentary or Top 10 video, plus captions, chapters, a description
with credits, three thumbnails, a QA report and a cost report. This file is
self-contained: you do not need any earlier conversation to use it.

## 1. What works, what is mock, what is stubbed

### Verified working in this build (mock mode, no API keys)

| Check | Result |
|---|---|
| A1 `python -m vidbush demo` (Panama Canal brief, 4 min documentary) | Passes every check: about 4.3 min, 40+ shots, 5 shot types, animated maps, bar chart, title cards, number cards, lower thirds, callouts, burned captions, ducked music, chapters, description with credits, 3 thumbnails, QA and cost reports. -14.0 LUFS, about -1.1 dBTP, 1920x1080, 30 fps, H.264 High, AAC 192k. |
| A2 script mode (`fixtures/demo_script.md`, Hoover Dam) | Passes the same checks. |
| A3 voiceover mode (`fixtures/demo_voiceover.mp3`, Erie Canal) | Passes the same checks. The script next to the mp3 is used for alignment. |
| A4 `python -m vidbush serve` | UI and API on port 8080. Create a project, watch stages and log, play the video, open a shot, swap it: only that segment re-renders. Checked in headless Chromium with no console errors. |
| A5 `revise demo "make the first 30 seconds faster and replace shot 7 with an AI image"` | Maps to `retime_section [0, 30] x0.7` then `change_type shot_007 -> ai_image`, applies both, re-renders only changed segments. |
| A6 `learn-style fixtures/reference_video.mp4 test_profile` | Finds 12 shots, the 2.39 letterbox and the caption band; writes `config/style_profiles/test_profile.yaml`. |
| A7 pytest | 750+ unit tests plus the slow demo integration test (`tests/test_integration_demo.py`), which runs the demo twice and asserts the second assembly renders 0 segments. |
| A8 crash and resume | `VIDBUSH_FAIL_AT=judge` stops the run at stage 8; `resume demo` skips intake through sourcing and finishes. |
| A9 Docker | Image builds (ubuntu 24.04 base), `docker compose up` serves the UI and API on 8080, and a project rendered inside the container. |
| A10 em dash | Zero in the repo; `tests/test_core_no_em_dash.py` guards it. |

Project status values are `done`, `needs_review` and `failed`. A project ends
`needs_review` when it finished but one of these happened:

- a role you configured as real fell back to mock;
- a writer section or director chunk fell back to heuristics while a real
  provider was configured (listed by scene id under "Heuristic fallbacks" in
  `qa_report.md`);
- a batch cost cap stopped the run (section 4b).

### What is mock in the demo (and why)

The build sandbox blocked every outside API (Wikipedia, YouTube, Commons,
Archive, LoC, NASA, OpenAI-style APIs, ElevenLabs, Hugging Face). So in the
demo:

- Research, outline, script, titles, description and the director's visual
  plan come from canned fixtures in `fixtures/llm/the_panama_canal/` (written
  to be accurate, with real Wikipedia sources).
- Footage and stills come from a bundled synthetic library
  (`fixtures/sample_assets/`), each visibly labelled SAMPLE FOOTAGE or SAMPLE
  IMAGE.
- The judge scores from image statistics, AI images are labelled gradients,
  and narration is espeak-ng (robotic but intelligible).
- Every mock call is marked in `cost_report.json` and listed in `qa_report.md`.

Motion graphics, captions, music, mixing, loudness, assembly, cache, editor,
chat revisions, thumbnails composition, chapters and the UI are all real code
paths, not mocks.

### Built but not verified against live services

All real adapters are written and unit tested against canned responses, but
were never run live because of the sandbox:

- OpenAI-compatible chat, vision and images (works with OpenRouter and other relays by base URL), fal.ai and Runware images, fal image-to-video.
- ElevenLabs TTS with timestamps, model listing, default premade voice.
- Tavily, Serper, Brave search; Wikipedia.
- YouTube (yt-dlp section downloads plus scene detection), Wikimedia Commons, Internet Archive, Library of Congress, NASA, Pexels, Pixabay, with per-host rate limits and on-disk search caching.
- faster-whisper transcription (model download was blocked).

Run `python -m vidbush doctor --probe` once your keys are in: it sends one tiny
request per configured role and tells you exactly which seat works.

Verified live after the build (internet granted): Archive, Library of Congress
and NASA search and download, YouTube search, Whisper transcription (a
voiceover with no script produced a full video). YouTube downloads need
cookies from servers and VPS IPs: set `YOUTUBE_COOKIES` or
`YOUTUBE_COOKIES_FROM_BROWSER` in `.env`. Set `VIDBUSH_CONTACT` to your email
so Wikipedia and Wikimedia do not rate limit you.

### Stubbed or partial

- DepthFlow: wired, off by default, untested (the CLI was not installable here). Ken Burns is used instead.
- ai_video seat: wired for fal-style image-to-video, off by default (0 percent in the mix), untested live.
- Remotion renderer (`renderer/`): renders the same motion spec and was tested on a title card, but the built-in Python renderer is the default and has the richer look.
- Optional SFX: a sparse synthesized whoosh at some motion graphic starts, off by default (enable with `sfx.json` in the music folder, see section 4).

## 2. First run on Windows 11 (PowerShell)

You received the code as `vidbush-source.zip` (and `vidbush.bundle`, the full
git history). Do every step in a normal PowerShell window (Start menu, type
PowerShell, press Enter). Paste one block at a time and wait for it to finish.

**Step 1. Install the tools.** Paste this whole block:

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Git.Git
winget install -e --id DenoLand.Deno
winget install -e --id OpenJS.NodeJS.LTS
```

Node is only needed if you want to rebuild the UI or try the Remotion
renderer. Deno is needed for YouTube downloads. Optional, for the free robot
narration in mock mode: install espeak-ng from
https://github.com/espeak-ng/espeak-ng/releases (the .msi file).

**Step 2. Close PowerShell and open a new window** so the new tools are on
your PATH. Check them:

```powershell
python --version
ffmpeg -version
```

**Step 3. Unpack the code to `C:\vidbush`.** Right-click
`vidbush-source.zip`, Extract All, choose `C:\`, so that the file
`C:\vidbush\pyproject.toml` exists. (Or with git:
`git clone C:\Users\<you>\Downloads\vidbush.bundle C:\vidbush`.)

**Step 4. Create the Python environment and install.** Paste as one block:

```powershell
cd C:\vidbush
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[whisper,dev]"
```

Every time you open a new PowerShell window later, run these two lines first:

```powershell
cd C:\vidbush
.\.venv\Scripts\Activate.ps1
```

**Step 5. Create your `.env`.**

```powershell
copy .env.example .env
notepad .env
```

For a mock test, leave it as is. For real videos fill the keys of the seat
policy (section 9): `WELLFLOW_CHEAP` and `WELLFLOW_API_KEY` (Wellflow),
`CLOSEROUTER_API_KEY`, optionally `ANTHROPIC_DIRECT_API_KEY`, the narration
keys `ELEVATE_API_KEY`, `ALGROW_API_KEY` and/or `ELEVENLABS_API_KEY`, the
image keys `IMAGE_API_KEY` and `IMAGE_FALLBACK_API_KEY` with `IMAGE_MODEL`
and `IMAGE_FALLBACK_MODEL`, and one of `SEARCH_API_KEY` (Tavily),
`PEXELS_API_KEY`, `PIXABAY_API_KEY`. Anthropic model ids are found by
Vidbush itself (section 9). Save and close. Older `.env` files that set
`LLM_PRIMARY_BASE_URL`, `LLM_PRIMARY_API_KEY`, `VISION_BASE_URL` and so on
are no longer read: copy those keys to the new names.

**Step 6. Check the setup, then run the demo** (takes 5 to 15 minutes):

```powershell
python -m vidbush doctor
python -m vidbush demo
```

The video is at `C:\vidbush\output\demo\final.mp4`. Open
`C:\vidbush\output\demo\qa_report.md` to see what was mock.

**Step 7. Start the editor UI.**

```powershell
python -m vidbush serve
```

Open http://localhost:8080 in your browser. Leave the PowerShell window open
while you use it (Ctrl+C stops the server).

**Step 8. Your first real project.** Fill the keys in `.env`, then:

```powershell
python -m vidbush doctor --probe
```

Every role you configured should say ok. Then create a brief file, for
example `C:\vidbush\briefs\my_first.md`:

```markdown
---
topic: The Johnstown Flood of 1889
target_minutes: 12
format: documentary
audience: older US viewers who love American history
angle: the dam the rich built and the town below it
---
```

and run:

```powershell
python -m vidbush run briefs\my_first.md --brand default
```

Or do the same from the UI (Projects, New project, Brief tab). A pasted
script works the same way (`python -m vidbush run my_script.md`), and so
does a finished voiceover (`python -m vidbush run narration.mp3`; put
`narration.md` with the script next to it for perfect alignment).

## 3. First run with Docker on a Linux VPS

```bash
git clone /path/to/vidbush.bundle vidbush
```

```bash
cd vidbush && cp .env.example .env && nano .env
```

```bash
mkdir -p output data music && docker compose up -d --build
```

```bash
docker compose logs -f vidbush
```

Open `http://<vps-ip>:8080` (put it behind your reverse proxy or an SSH
tunnel: there is no login, it is a single-operator tool). Output, data, your
music and your brands and style profiles are bind mounted, so they survive
rebuilds. Run CLI commands inside the container, for example:

```bash
docker compose exec vidbush python -m vidbush doctor --probe
```

```bash
docker compose exec vidbush python -m vidbush demo
```

The image includes FFmpeg, espeak-ng, deno (for YouTube) and faster-whisper.
The whisper model (about 150 MB) downloads on first use.

## 4. Where things go

- **Music:** any folder of mp3, ogg, wav or m4a files. Point a brand at it
  (`music_folder` in the brand yaml, or `MUSIC_FOLDER` in `.env`). In Docker,
  use `./music` on the host, which is `/app/music` inside. Add a `tracks.json`
  for mood matching: `{"track.mp3": {"mood": "tense", "energy": 0.7, "loop": true}}`.
  Moods used: calm, tense, uplifting, mysterious (hook and climax get uplifting or tense, setup and resolution calm, turns tense). Without it
  tracks are chosen at random (logged). Optional `sfx.json` in the same folder:
  `{"enabled": true, "max_per_minute": 3, "gain_db": -24}`.
- **Brands:** `config/brands/<id>.yaml` (or the Brands page). Fields: voice
  (ElevenLabs voice_id, model `auto`), style_profile_id, music_folder,
  intro_path and outro_path (video files), blocklist (domains), niche_notes
  (read by the writer and director, put your channel voice and rules here),
  language. Use with `--brand <id>`.
- **Style profiles:** `config/style_profiles/<id>.yaml`. Copy
  `documentary_default.yaml` and change the mix, pacing, captions, letterbox,
  music ducking and Ken Burns intensity. `learn-style <url or file> <name>`
  writes one from a reference video.
- **Your own footage library:** set `sources.local_library` in
  `config/vidbush.yaml` to a folder of clips and images. Optional sidecar
  `<file>.json` with `title`, `tags`, `licence`, `author` improves matching.
- **Blocked domains and YouTube:** `sources.blocklist` and
  `sources.allow_youtube` in `config/vidbush.yaml`, or `YOUTUBE_ENABLED` in `.env`.
- **Outputs:** `output/<project_id>/`. Logs: `data/logs/<project_id>.jsonl`.

## 4a. Visual mix, card spacing and director context

The style profile splits animation into two numbers in `visual_mix`:

- `full_screen_cards`: title, number, chart or map as a full frame.
- `animated_overlays`: maps, charts, callouts, arrows and lower thirds moving
  over footage or a still. They are counted on top of the shot they decorate.

Documentary default: `{footage: 20, footage_overlay: 21, image: 7,
image_overlay: 10, full_screen_cards: 12, animated_overlays: 30}`. That turns
into these shot targets: footage 20, footage with overlay 42.2, image 7,
image with overlay 18.8, full screen cards 12 (percent of shots).

A map or chart over the picture is drawn as a framed inset panel in the
upper right. `pacing.card_spacing_s: 45` allows at most one free full screen
card per 45 seconds outside the hook and the recap (0 turns it off). Top 10
number cards always stay. Old profiles with a `motion_graphic` key still
load; that number is read as full screen cards. `qa_report.md` has an
Animation table: cards, shots with an overlay, and map and chart insets.

Every director call sees:

- a one paragraph summary of the whole script, cached in
  `work/script_summary.json`;
- the ideas chosen for the previous two chunks;
- the visual anchors used so far;
- the next beat's narration.

Director calls therefore run one after another, not in parallel.

## 4b. Batch queue (many videos overnight)

Put briefs (`.md` with a `topic` front matter), scripts (`.md` or `.txt`)
and voiceovers (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`) in one folder and run
them one after another for one brand. A script with the same name as a
voiceover (`narration.mp3` plus `narration.md`) is that voiceover's script,
not a separate video. Sub folders are not read. Instead of a folder you can
pass a list file: one path per line, blank lines and `# comments` ignored,
relative paths are relative to the list file.

Windows (PowerShell):

```powershell
python -m vidbush batch briefs --brand default
python -m vidbush batch briefs\tonight.txt --brand history --cost-cap 3 --id-prefix ep_
python -m vidbush batch-status
python -m vidbush batch-status <batch_id>
python -m vidbush batch-pause <batch_id>
python -m vidbush batch-resume <batch_id>
```

Docker:

```bash
docker compose exec vidbush python -m vidbush batch /app/data/briefs --brand default
```

```bash
docker compose exec vidbush python -m vidbush batch-status
```

(Put the folder somewhere the container sees, for example `./data/briefs`
on the host, which is `/app/data/briefs` inside.)

- Optional flags of `batch`: `--profile`, `--format`, `--minutes` (same as
  `run`), `--cost-cap <usd>`, `--id-prefix <text>`, `--batch-id <id>`,
  `--plan` (queue and print the plan, do not run), `--quiet`. Project ids
  come from the file names like `run` (a second file with the same name gets
  `_2`). An existing project with the same id is replaced, like `run`.
- **Cost cap:** each project run stops once its cost (from
  `cost_report.json`) passes the cap, default `batch.cost_cap_usd: 5.0` in
  `config/vidbush.yaml` (`--cost-cap 0` or `null` in the config means no cap).
  The check happens between stages, so a run stops at the next stage
  boundary and can spend a little more than the cap inside the stage that
  crossed it. The project ends as `needs_review` with the reason in
  `project.json` (`status_note`), `work/cost_cap.json` and the log, and the
  batch shows it as `capped`. Finish it later with
  `python -m vidbush resume <id>` (no cap) or `resume <id> --cost-cap 10`.
  `batch-resume <batch_id> --retry-failed --cost-cap 10` requeues every
  failed and capped project of a batch with a higher cap.
- **Failures** do not stop the batch: the error is recorded and the next
  project starts.
- **Pause** takes effect between projects: the project that is running
  finishes, then the batch stops. `batch-resume` continues from the first
  unfinished project; a project that was interrupted (Ctrl+C, crash, server
  restart) resumes from its own first unfinished stage.
- **Report:** `data/batches/<batch_id>/batch_report.md` (and
  `batch_report.json`), one row per project: id, input, mode, status
  (queued, running, done, needs_review, capped, failed, skipped), duration,
  total cost, cost per minute, QA percent above the judge threshold, note.
  The path is printed at the end.
- **UI:** the Queue page (http://localhost:8080/#/queue) shows the same
  table, refreshes itself, and has New batch (a folder or list file path on
  the machine that runs Vidbush), Pause and Resume. A batch started from the
  UI runs in the server's worker and continues after a server restart.

## 5. Cost expectations per finished minute (real providers)

The demo made these calls for 4.3 minutes of video: 1 research, 1 outline,
about 8 writer calls, director calls per scene, 237 judge calls (each with up
to 3 low-detail keyframes), 90 QA vision calls, 35 TTS chunks (3,531
characters, about 825 characters per minute), and image calls only for
fill-in shots. The demo itself cost $0 (all mock). Estimated real cost per
finished minute:

| Item | Volume per minute | Typical price | Per minute |
|---|---|---|---|
| ElevenLabs narration | about 825 characters | $0.15 to $0.30 per 1k chars | $0.12 to $0.25 |
| Judge (vision) | about 55 calls, 1 high detail and 2 low detail keyframes each (fewer after the local pre-filter) | cheap vision model to premium | $0.03 to $0.45 |
| QA pass (vision) | about 21 calls, 1 image each | same | $0.01 to $0.05 |
| Research, script, director, editor (text) | about 15k to 30k tokens | mid to strong model | $0.03 to $0.15 |
| AI images | 0 to 1 fill-in images | $0.02 to $0.04 each, ceiling $0.05 | $0.00 to $0.04 |
| Web search, stock APIs | a few queries | Tavily or free tiers | under $0.01 |
| **Total** | | | **about $0.20 to $0.80** |

Narration and the judge dominate. A cheap vision model for `judge` and
`vision` keeps you near the low end. Every real run writes exact numbers
to `cost_report.json` (`per_minute_usd`); use those after your first video.
The judge line rose from 3 low detail images per call to 1 high plus 2 low
(about 255 to 595 image tokens per call). Set `judge.first_keyframe_detail: low`
to go back to the old cost. When a provider returns no cost, the built-in
estimate prices low images at `cost.vision_per_image_usd` (0.002) and high or
auto images at `cost.vision_per_image_high_usd` (0.010), so it reads high on
purpose.

## 6. Seats: what to map each role to

Roles live in `config/seats.yaml`. Each role lists providers in order; the
first configured one is used and only a failed request moves to the next.
If every real provider for a role fails, the stage fails (resume later)
unless you set `allow_mock_fallback: true`. Model names below are guidance;
pick whatever is best and cheapest this month.

| Role | Used for | Map it to | Why |
|---|---|---|---|
| researcher | fact cards, anchors, fact check | a strong reasoning model with long context | accuracy matters most here; few calls |
| writer | outline and script, titles, description | your best writing model | quality of narration is the product |
| director | visual ideas per scene, motion graphic specs | a strong model that follows JSON schemas well | one call per scene, structured output |
| editor | chat revisions, beat rewrites | a cheap fast model | short prompts, maps text to fixed ops |
| judge | scoring candidates (many calls) | a cheap vision model | volume; "good enough" bar |
| vision | post-render QA, thumbnails | same as judge | volume |
| cheap_classifier | learn-style shot classification | the cheapest vision model | one call per shot |
| image | AI fill-in shots and thumbnail backgrounds | one consistent photographic model under $0.05 | never mixes styles within a video unless the primary is down |
| video | optional image-to-video | fal-style model | off by default |
| tts | narration | Elevate, then Algrow, then ElevenLabs (section 9) | Elevate and Algrow audio is aligned to the script by faster-whisper; ElevenLabs returns word timestamps |
| search | research web search | Tavily, Serper or Brave | any one is enough |

Providers marked down after transport failures are retried every 10 minutes.
429 rate limits back off using Retry-After and never mark a provider down.

## 7. Known gaps and the next three improvements

Known gaps:

- No live provider has been exercised (sandbox). Expect small request-shape
  fixes on first contact; the per-model quirk learner handles the common ones.
- The heuristic fallbacks (used only when no LLM answers) give thin scripts
  and generic shot intents. With a real LLM they are not used. If one is
  used anyway, QA names the scene and the project ends needs_review.
- Director calls now run in sequence (they carry the previous ideas), so the
  shot list stage takes longer on long scripts than before.
- The map and chart inset look was checked on single frames only. Judge it
  on a real render and tune `INSET_FRACTION` in `vidbush/mg/inset.py` if
  needed.
- The bundled music is generated by code (CC0) and only checked by metrics,
  not by ear. Use your own library for real videos.
- The local pre-filter (section 8) catches burned-in subtitles, static corner
  logos and talking heads; semi-transparent watermarks over moving content
  still need the vision judge.
- No login on the UI; keep port 8080 private.

Next three improvements, in order:

1. **First live run and calibration:** set keys, `doctor --probe`, run one
   real 10 minute project, then tune the judge threshold, director prompt
   and pacing from that QA report. This is where most of the quality jump
   comes from.
2. **Calibrate the footage filters on real footage:** the pre-filter,
   YouTube window selection and dedupe rule (section 8) were tested only on
   synthetic frames and canned yt-dlp responses. Check the prefilter log line
   and the rejects in qa_report after the first real run and tune
   `judge.prefilter` thresholds.
3. **Batch and schedule:** a queue page that takes a list of briefs per brand
   and renders overnight, with a per-run cost cap.
   The queue, cost cap and Queue page are now built (section 4b); what is
   left is a schedule (start a batch at a set time).

## 8. Footage quality and YouTube options

All options below live in `config/vidbush.yaml` unless a different file is named.
Defaults are shown; nothing here blocks footage except the blocklist.

**Local pre-filter** (`judge.prefilter`, code in `vidbush/judge/prefilter.py`).
Runs on keyframes before any vision call; a rejected candidate never costs a
vision call and shows a reason such as `burned_in_captions (prefilter: ...)`,
`channel_logo (prefilter: ...)` or `talking_head (prefilter: ...)`.

| Option | Default | Meaning |
|---|---|---|
| enabled | true | master switch |
| text, logo, face | true | switch each check |
| text_min_keyframes | 2 | keyframes with subtitle-like text before a reject |
| text_band_top | 0.62 | share of frame height where the subtitle band and lower third start |
| text_min_width | 0.25 | a text line in the band must span this share of the width (a small corner word never counts) |
| text_min_confidence | 0.6 | OCR confidence floor |
| logo_corner | 0.22 | corner box size, share of width and height |
| logo_min_area, logo_max_area | 0.002, 0.04 | size range of a stable corner mark, share of the frame |
| logo_max_diff | 6.0 | pixel difference that still counts as identical across keyframes |
| face_min_area | 0.06 | a frontal face box this large (share of frame) is a talking head candidate |
| face_centre | 0.5 | face centre must sit in this middle share of the frame |
| face_min_share | 0.5 | reject when more than this share of keyframes show such a face |

OCR uses rapidocr-onnxruntime (weights inside the wheel) and faces use the
OpenCV haar cascades, so OpenCV is pinned to 4.14.0.94 (5.x dropped the
cascade files). Local library, uploads and generated media are never
pre-filtered; logo and face checks run on video only. A missing package
disables its check with one warning. Each judge pass logs one `prefilter` line
with counts.

**Judge image detail** (`judge`): `first_keyframe_detail: high` and
`keyframe_detail: low` set the image detail of the first and the other
keyframes of each judge call (high, low or auto). Post-render QA, thumbnails
and learn-style always send low. `cost.vision_per_image_high_usd: 0.010`
prices high and auto images in the estimate when a provider returns no cost.

**YouTube** (`sources`, code in `vidbush/source/youtube.py`):

| Option | Default | Meaning |
|---|---|---|
| flag_standard_youtube | true | Standard YouTube Licence clips get the flag `standard_youtube_licence`, listed in qa_report (`standard_youtube_assets`). Flag only. |
| youtube_cookies_from_browser | "" | same as yt-dlp `--cookies-from-browser`, e.g. `firefox` or `"chrome:Profile 1"` |
| youtube_cookie_file | "" | Netscape cookies.txt; relative paths resolve against the repo root |
| youtube_download_workers | 3 | parallel downloads of different videos; searches stay serial at 0.5 per second |
| youtube_windows | 3 | windows sampled across a video when no transcript or chapter matches the shot |
| query_budget.youtube | 250 | searches per run; qa_report shows "YouTube searches used: X of N" and a warning when a budget ran out |

Brand option (`config/brands/<id>.yaml`):

```yaml
licence_policy:
  prefer_creative_commons: true   # first YouTube search uses the Creative Commons filter, then the normal search
```

With a transcript or chapter list, the window whose text best matches the
shot's intent and must_show is downloaded; otherwise up to `youtube_windows`
short windows across the video, each with 10 s padding. Downloads prefer
h264 so scene detection can decode them.

**Selection dedupe** (no options): the same video segment is never picked
twice in one scene or for two consecutive shots (a motion graphic between
them does not reset this). In another scene a reuse costs 40 score points.

**LLM providers** (`config/seats.yaml`, `providers.<name>.options` or a role
entry's `options`): `max_tokens` is an optional output cap, unset by default
(no cap sent); `timeout_s` replaces the role timeout (writer 600 s, other LLM
roles 180 s). A reply cut off by the length limit is continued once and
joined; if it is cut off again the request moves to the next provider.

## 9. Reviewing a render

After every real render, review it in the tool instead of a voice note. The
review becomes structured data the next tuning session reads.

**In the UI.** Open the project and press **Review** (or go to
http://localhost:8080/#/review/<project_id>). The video plays at the top; below
it is one row per shot with its thumbnail, start time (click to seek), type,
intent, selected source (site and licence), judge score and flags, and the
reasons its rejected candidates were turned down. For each shot pick a verdict
(good, wrong picture, wrong timing, quality problem, other), tick any hard
reject (burned-in text, watermark or logo, talking head, anachronism, repeated
clip) and write a comment. Comments have no length limit. The header shows
"n of m reviewed" and has a project comment box. Everything saves by itself.

Keys: Left and Right move between shots (the player seeks there), 1 to 5 set
the verdict (press again to clear), c jumps to the comment, Space plays or
pauses, Esc leaves a text box. The project page shows "Reviewed: n of m".

**Apply fixes** previews the edits first, like chat revisions: wrong picture
re-sources the shot with your comment added to its must_avoid and search
queries (an AI image is regenerated with your comment as an avoid note);
quality problem with a hard reject ticked re-sources with that reject added
to must_avoid; wrong timing gets an "Open duration editor" link. Everything
else stays a comment. Only changed segments re-render.

**Files.** `output/<id>/review.json` (one object per shot plus the project
comment, with timestamps) and `output/<id>/review.md` (Export button: verdict
counts, hard rejects with timestamps, then every commented shot in order;
hand this file to the next tuning session).

**CLI** (Windows: same commands in PowerShell; Docker: prefix with
`docker compose exec vidbush`):

```bash
python -m vidbush review-summary <id>
```

```bash
python -m vidbush review-export <id>
```

```bash
python -m vidbush review-calibrate <id>
```

```bash
python -m vidbush review-fixes <id>
```

`review-fixes <id> --apply` applies them and re-renders. `review-set <id>
<shot> --verdict wrong_picture --comment "..." --reject watermark_logo` and
`review-comment <id> "..."` do what the page does (`--comment -` reads stdin).

**Calibration.** `review-calibrate` lists every shot you marked wrong picture
or quality problem that the judge scored at or above `judge.threshold`, every
shot you marked good below it, a suggested threshold (fewest disagreements)
and the hard rejects the judge missed most often, each with the
`judge.prefilter` setting to tune. Wrong timing and other are left out: they
say nothing about the judge. Change `judge.threshold` and `judge.prefilter` in
`config/vidbush.yaml` from this output after each run.

## 10. TTS seats, Anthropic provider, streaming and seat policy (new)

Everything in this section was built and tested offline against canned
replies taken from RadioPlay's working clients. Check it live with the probes
in 10f before the first real run.

### 10a. Seat policy (Adam, 2026-11-05)

`config/seats.yaml` now holds this policy:

| Role | Primary | Then | Then |
|---|---|---|---|
| researcher, director | `wellflow_cheap` Sonnet 5.5 | `anthropic_direct` Sonnet 5.5 | `closerouter` Sonnet 5.5 |
| writer | `wellflow_cheap` Opus 5.5 | `anthropic_direct` Opus 5.5 | `closerouter` Opus 5.5 |
| editor, judge, vision, cheap_classifier | `closerouter` (Gemini 3.8 Flash) | `wellflow` (normal key) | |
| tts | `elevate_tts` | `algrow_tts` | `elevenlabs` |
| image | Elevate | Algrow | |

- `wellflow_cheap` uses the Wellflow key named `WELLFLOW_CHEAP`. It serves
  Anthropic models only, so it sits only on the three Anthropic roles.
- Anthropic seats carry `model_match: {family, version}` instead of an id.
  At the first use Vidbush reads the provider's `GET /models`, picks the id
  for that family and version (`claude-opus-5-5` on Wellflow,
  `anthropic/claude-opus-5.5` on CloseRouter), and logs the match once
  ("model resolved from the provider model list"). No match means that seat
  is skipped with a warning and the next one serves. To pin an exact id on
  `wellflow_cheap`, set `LLM_WRITER_MODEL`, `LLM_DIRECTOR_MODEL` or
  `LLM_RESEARCHER_MODEL`.
- Primary switch: `VIDBUSH_PREFER_PROVIDER=anthropic_direct` moves that
  provider to the front of every role it serves, until
  `VIDBUSH_PREFER_UNTIL` (UTC) passes. `.env.example` is set for Adam's plan:
  `anthropic_direct` for 48 hours from 2026-11-05, then `wellflow_cheap`
  again with no edit. Clear both lines to stop early.
- `elevate_tts` is the type the laptop lane adds in
  `vidbush/providers/tts_elevate.py`. Until it is registered, the entry is
  skipped with one warning ("provider type not registered, entry skipped")
  and Algrow narrates.

### 10b. TTS seats

- **Registry.** TTS types are listed in `vidbush/providers/seats.py`
  `TTS_TYPES`, one line per type (`"algrow_tts": tts_algrow,`). A new
  adapter registers with one import and one line. The value can be the module
  (its `synthesize` is called) or the function itself.
- **Algrow** (`algrow_tts`, `vidbush/providers/tts_algrow.py`): submit to
  `/api/generate-simple`, poll `/api/job-status/<id>` every 3 s up to
  `timeout_s` (600), download the hosted MP3. Model `eleven_v4`. A refused
  model moves the whole run to `fallback_model` (eleven_multilingual_v2).
  Algrow is a subscription: each call shows 0 USD in `cost_report.json` and
  the reported credits in the entry's `credits` field. Scripts under 200
  characters are padded with break tags (Algrow's floor).
- **Voice ids per provider.** Brand yaml: `voice.voice_id` is the default,
  `voice.voice_ids` maps a tts provider name to its own id, for example
  `voice_ids: {algrow_tts: "JBFqnCBsd6RMkjVDRZzb"}`.
- **Alignment routing.** Captions need word timings. A TTS reply with
  timings that agree with the audio (ElevenLabs) is used as is. A reply
  without timings (Algrow, Elevate) is aligned against the script with
  faster-whisper, the same path a voiceover with a script uses. Install the
  whisper extra (`pip install -e ".[whisper]"`); without it the timing
  falls back to silences and a warning says so. If one provider's timings
  turn out unreliable, set `force_align: true` in its `options` in
  `config/seats.yaml`. The method is in `work/narration.json`
  (`aligned_by`) and in the log line "narration ready".
- Chunking, the 40 ms crossfade and -16 LUFS narration are unchanged. The
  first chunk is now made before the others, so a refused model switches the
  run before any other chunk exists.

### 10c. Native Anthropic provider

Provider type `anthropic` (`vidbush/providers/llm_anthropic.py`) calls the
Messages API directly (`POST /v1/messages`, `x-api-key`,
`anthropic-version: 2026-10-21`). It behaves like `openai_compat`: the same
retries, 429 backoff, fallback, one continuation of a cut reply, one retry of
unparseable JSON, and quirks learned from 400s. Differences:

- the system prompt is the top-level `system` field; JSON is asked for in it;
- `max_tokens` is always sent (64000 streamed, 16000 not, or
  `options.max_tokens`);
- `temperature` is never sent (current Claude models reject it);
  `options.send_temperature: true` sends it for an older model;
- `options.effort` (`low` to `max`) is sent as `output_config.effort`;
- cost comes from a built-in price table (Opus 5.5 4 / 20 USD per million,
  Sonnet 5.5 2 / 10, Haiku 5.5 0.10 / 0.50, cache reads and writes at their
  own rates); override with `options.price_per_mtok: {prompt, completion,
  cache_read, cache_write_5m}`.

### 10d. Streaming default

Every `openai_compat` provider now streams unless its `options.stream` is
`false` (Adam's rule: Wellflow and CloseRouter are faster and stop timing
out when streamed). The stream reader skips `: ping` keepalives, takes usage
from the final chunk, drops a stream that stops without `[DONE]` or a finish
reason (it is retried, never used half), accepts a relay that answers plain
JSON, and retries without `stream_options` if a relay rejects it.

### 10e. Source budgets

Run budgets (`sources.query_budget`, Pexels 150, YouTube 250, Pixabay 600,
loc 300, others 800) are now shared fairly. Every shot that still needs
candidates gets an even share; queries go out in rounds (first query of every
shot, then the second for shots still short, then the third); a shot whose
web results already fill its plan stops searching and its unused share goes
to the others. `qa_report.md` shows one line per budgeted source, for example
`Pexels searches used: 148 of 150 across 52 shots, 9 deferred by fair shares`,
and a warning when a budget ran out.

### 10f. Live checks for the laptop

Run these after filling `.env`. Each sends one or two small requests.

```powershell
python -m vidbush doctor --probe
python -m vidbush.live_probe seats
python -m vidbush.live_probe anthropic
python -m vidbush.live_probe stream wellflow_cheap
python -m vidbush.live_probe stream closerouter
python -m vidbush.live_probe stream wellflow
python -m vidbush.live_probe algrow --credits
python -m vidbush.live_probe algrow
```

What to look for: `seats` lists `wellflow_cheap:claude-opus-5-5` for the
writer and `claude-sonnet-5-5` for director and researcher (with
`anthropic_direct` first while the preference is on); `anthropic` prints
`{"ok": true}` twice with a cost above 0; each `stream` prints a value in a
few seconds; `algrow` writes `data/live_probe/algrow.mp3`, says
`words_returned: false`, shows credits, and `aligned_by: whisper` with
word times that look right when you play the file.

See also: `docs/ARCHITECTURE.md` (stage graph, contracts, cache rule),
`docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/BLOCKERS.md`, `docs/SPEC.md`.
