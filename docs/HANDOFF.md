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
- a cost cap stopped the run (`run --cost-cap`, `resume --cost-cap` or a
  batch cap; sections 4b and 12).

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
policy (section 10a): `WELLFLOW_CHEAP` and `WELLFLOW_API_KEY` (Wellflow),
`CLOSEROUTER_API_KEY`, `OPENLUX_API_KEY`, `ANTHROPIC_KEY`, the narration
keys `ELEVATE_API_KEY`, `ALGROW_API_KEY` and/or `ELEVENLABS_API_KEY`, the
image keys `IMAGE_API_KEY` and `IMAGE_FALLBACK_API_KEY` with `IMAGE_MODEL`
and `IMAGE_FALLBACK_MODEL`, and one of `SEARCH_API_KEY` (Tavily),
`PEXELS_API_KEY`, `PIXABAY_API_KEY`. Anthropic model ids are found by
Vidbush itself (section 10a). Keys may also live in `.env.local`, which wins
over `.env` (the process environment wins over both). Save and close. Older `.env` files that set
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

`run` refuses to start when a project with the same id already exists. Use
another `--id`, continue it with `resume <id>`, or pass `--fresh` to delete
it and start again (`demo` always starts fresh). `--cost-cap <usd>` stops the
run once the project spent that much (section 12).

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
  Tile audit of a project: `output/<project_id>/audit/`. Stored model replies:
  `reply_cache/` beside the source cache (section 12).

## 4a. Visual mix, card spacing and director context

The style profile splits animation into two numbers in `visual_mix`:

- `full_screen_cards`: title, number, chart or map as a full frame.
- `animated_overlays`: maps, charts, callouts, arrows and lower thirds moving
  over footage or a still. They are counted on top of the shot they decorate.

Documentary default: `{footage: 20, footage_overlay: 21, image: 7,
image_overlay: 10, full_screen_cards: 5, animated_overlays: 37}`. That turns
into these shot targets: footage 20, footage with overlay 47.2, image 7,
image with overlay 20.8, full screen cards 5 (percent of shots).

The overlay shares are a ceiling, not a quota. On-screen text never repeats
the narration (the captions already show it): a callout or text overlay adds a
number, date, place or role label, unit conversion or source, or says the
point in other words, and it appears 0.2 s after the narrator says its anchor
words (or at the next shot's start when the word fell in the last second
before a cut), holding at least 2.5 s (about 0.3 s per word). An overlay slot with
nothing honest to show stays plain footage, and a card slot whose beat has
nothing to draw (or whose card was already shown) becomes footage too.
`qa_report.md` shows the on-screen text line: verbatim share and the median
gap between each text overlay and its words.

A map or chart over the picture is drawn as a framed inset panel in the
upper right. `pacing.card_spacing_s: 75` allows at most one free full screen
card per 75 seconds outside the recap, the hook included (a top 10 hook
stays exempt; 0 turns it off). Top 10 number cards always stay. Old profiles
with a `motion_graphic` key still load; that number is read as full screen
cards.

Card length and timing (shots/rules.py):

- `pacing.card_hold_max: 4` caps a static card (title, text, quote,
  callout, number); a full screen map or chart holds at most 7 s. The excess
  goes to a neighbour, and what is left becomes footage of the same beat.
- A text or number card must start on its cue: the word that speaks its
  number or the first word of its title starts at most 0.5 s before the card
  or 1.5 s into it. Otherwise the card moves to the cue, or becomes an
  overlay on footage.
- A card that loses its slot (spacing, cue) keeps its message on the
  picture: a title as a lower third, a number or text card on a transparent
  background, a map or chart as an inset.
- Chapter titles are lower thirds on footage, not full screen cards. Call to
  action beats get no card at all.

Pictures, maps and labels follow the words (run 3, shots/assembler.py,
shots/rules.py, mg/mapdraw.py):

- Every footage and still idea names its `anchor`, the narration words it
  illustrates. A picture slot shows the idea whose words are spoken in it
  (either family; the slot keeps its type, so the mix holds). The log line
  "pictures placed on their words" gives `anchor_misses`.
- A map or chart, inset or full screen, appears 0.2 s after its first place
  or number is spoken and stays at least 4 s, or moves or is dropped. At most
  one map or chart per 15 s, the same map not within 180 s, and a map only
  when it shows a new place (60 s window), on every path to the screen.
- Overlays go where their words are, not only to planned overlay slots, up
  to the profile's overlay share.
- A map's title is built from what it draws. Title, credit, locator and all
  labels share one collision check; a close-up map (zoom 5 or more) gets a
  small locator map. Small US towns resolve from USGS GNIS when a map names
  them (`assets/geo/gnis_places.json.gz`, rebuild with
  `E:\coding projects\Vidbush\geo_tools\build_gnis.py`).
- No free highlight boxes or arrows. A lower third must name something in the
  shot's `must_show` (chapter titles too), no country or decade
  subtitles, one label per name per minute. The hook title card is dropped
  when it repeats the opening line.

`qa_report.md` has an Animation table: cards, shots with an overlay, map and
chart insets, full screen card time (percent of runtime) and the longest text
or number card hold. Acceptance checks `card_time_sparing` (at most 10
percent of the runtime, skipped for top 10) and `card_hold_short` (at most
`card_hold_max` plus 0.5 s).

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
  `_2`). An existing project with the same id is replaced (a batch always
  replaces its own projects; `run` refuses unless `--fresh` is given).
- **Cost cap:** each project run stops once its cost (from
  `cost_report.json`) reaches the cap, default `batch.cost_cap_usd: 5.0` in
  `config/vidbush.yaml` (`--cost-cap 0` or `null` in the config means no cap).
  The cap is checked before every live paid call, so it holds inside a stage.
  With several calls in flight a run can pass the cap by at most the calls
  already sent. A stage stopped inside stays pending, and the resume reruns
  it with the replies already paid for served free from the reply cache. The project ends as `needs_review` with the reason in
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
  total cost, cost per minute, pre-render judge percent (bookkeeping, not a
  quality measure), audit noticeable percent (when the tile audit ran), note.
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
| tts | narration | ElevenLabs, then Elevate, then Algrow (section 10a) | Elevate and Algrow audio is aligned to the script by faster-whisper; ElevenLabs returns word timestamps |
| search | research web search | Tavily, Serper or Brave | any one is enough |
| auditor | tile audit of a finished video (section 12) | Sonnet 5.5 (CloseRouter, then openlux, then Anthropic direct) | agreed best with the reference tiles; Gemini 3.8 Flash under-flagged, Opus cost 2.7 times more |

Providers marked down after transport failures are retried every 10 minutes.
An error about the seat itself (HTTP 401, 402 or 403, model_not_in_channel,
usage or credit limits) marks that provider and model down at once, until the
same 10 minute probe; an error about one request only falls back for that request.
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

**AI stills** (`judge.ai_retries: 2`): every generated still is judged against
the narration, intent and must_show before use. A still below the threshold
is regenerated with "Must clearly show: ..." up to `ai_retries` times, and
the best take wins. The judge's reason goes to the editor rewrite as a
"Problem:" line only, never into the prompt or the negative prompt. A mock
judge score never triggers a retry. Each regeneration is a cost entry with the
note `image retry`; the per image ceiling still applies. The regenerate_ai
edit op uses the same loop and always selects the new batch (old takes stay
in candidates). A shot whose subject is a text artifact (a newspaper
headline, a labelled map or a map of a route, the text of a letter, a web
page, sheet music, a chart of data) is never converted to an AI still: it
keeps its best sourced candidate and gets the flag `needs_motion_graphic`
(listed in qa_report). Mock fallbacks during titles and thumbnails are added
to qa_report `mock_used` after package.

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
The reply cache key ignores `max_tokens`: after changing an output cap, clear
the reply cache folder or run with `--no-reply-cache` (section 12).

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
re-sources the shot and rejects the reviewed pick. Your comment is split in
two. The wish ("I want steam shovels" gives "steam shovels") becomes the
shot's `operator_wish`, which the judge reads, and a search query. The
complaint sentences ("this shows a modern ship") go to must_avoid. The whole
comment is never written into must_avoid. An AI image is regenerated with the
wish as the prompt ("Show: ...") and the complaint as the negative prompt;
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

### 10a. Seat policy (Adam, 2026-11-11)

`config/seats.yaml` holds this policy, in this exact order:

| Role | Primary | Then | Then |
|---|---|---|---|
| writer (Opus 5.5) | `wellflow_cheap` `claude-opus-5-5` | `closerouter` `anthropic/claude-opus-5.5` | `anthropic_direct` `claude-opus-5-5` |
| director, researcher (Sonnet 5.5) | `closerouter` (model_match sonnet 5.5) | `openlux` `claude-sonnet-5-5` | `anthropic_direct` `claude-sonnet-5-5` |
| editor (Grok 4.7, text only) | `closerouter` `x-ai/grok-4.7` | `openlux` (model_match grok 4.7) | `wellflow` (model_match grok 4.7) |
| judge, vision, cheap_classifier (Gemini 3.8 Flash) | `closerouter` `google/gemini-3.8-flash` | `openlux` (model_match gemini 3.8-flash) | `wellflow` `gemini-3.8-flash` |
| tts | `elevenlabs` (eleven_v4) | `elevate_tts` (eleven_v4) | `algrow_tts` |
| image | Elevate | Algrow | |

- Keys: `WELLFLOW_CHEAP` (the Wellflow key that serves Anthropic models
  only; it answers 403 `model_not_in_channel` for Sonnet, so it sits only
  on the writer), `WELLFLOW_API_KEY`, `CLOSEROUTER_API_KEY`,
  `OPENLUX_API_KEY` (base `OPENLUX_BASE_URL`, default
  `https://api.openlux.ai/v1`), `ANTHROPIC_KEY` (Adam's own key, native
  Messages API). Vidbush reads `.env` and then `.env.local` (`.env.local`
  wins; the process environment wins over both).
- The editor is Grok 4.7 and text only: chat revisions, beat rewrites and the
  AI image prompt safety rewrite, the role that must not refuse or water down
  a prompt. The image roles stay on the cheap vision model.
- Exact ids are used where they are verified. Where a relay's spelling is not
  known the entry names `model_match: {family, version}` (plus `prefix: ""`
  for non Claude families such as grok and gemini). At the first use Vidbush
  reads the provider's `GET /models`, picks the id for that family and
  version, and logs the match once ("model resolved from the provider model
  list"). No match means that seat is skipped with a warning and the next one
  serves. `LLM_DIRECTOR_MODEL` and `LLM_RESEARCHER_MODEL` pin CloseRouter's
  Sonnet id; a pin that names another family or version than the entry's
  model_match (for example an old `anthropic/claude-sonnet-5`) is ignored with
  one warning ("model_env pin does not match model_match, ignored"). Exact id
  entries read no env pin, so old `.env` lines such as `LLM_WRITER_MODEL`
  change nothing.
- Prices without a cost in the reply: `options.price_per_mtok` on the role
  entry (wellflow_cheap Opus flat 0.30 USD per million, openlux Sonnet 0.1765
  in and 0.8824 out, CloseRouter Opus 0.205866 in and 1.029332 out), else the
  provider's own price list, else the generic estimate.
- A seat that cannot serve at all (HTTP 401, 402, 403, `model_not_in_channel`,
  usage or credit limits) is marked down for 10 minutes at the first such
  reply, so later requests go straight to the next seat (section 6).
- Primary switch: `VIDBUSH_PREFER_PROVIDER` moves the named providers to the
  front of every role they serve (tts and image too) until
  `VIDBUSH_PREFER_UNTIL` (UTC) passes. `.env.example` leaves both empty.

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
- **Elevate** (`elevate_tts`, `vidbush/providers/tts_elevate.py`, ported
  from RadioPlay): submit to `POST /v2/media` (`{type: tts, prompt, voice_id,
  model_id}`), poll `/v2/media/<id>?type=tts` every 18 s, download the hosted
  MP3. The request timeout is 400 s plus 0.002 s a character, capped at
  `timeout_s` (600). A 429 for the concurrent limit is waited out, and a task
  still running at its deadline is resumed by the retry instead of paid twice.
  Subscription: 0 USD per call. `doctor --probe` reads `/v2/media/usage`
  with the key, then `/v2/media/health`.
- **The model that spoke** is returned by every TTS adapter and logged once per
  run ("tts model", provider, model). Each narration chunk gets the previous
  and next chunk's text (`previous_text`, `next_text`); ElevenLabs sends them
  for models that take them (eleven_v4), the job APIs ignore them.
- **One narrator per video.** A cached chunk counts only when the tts seat that
  serves now (`Seats.tts_engine`) made it. When a seat goes down mid run, the
  chunks from the other provider are made again on the serving seat; still
  mixed raises ProviderError. A re-narrated beat (ops rewrite_beat) from another
  provider than the narration's raises, and it gets its neighbours' text too.
- **Spent balance on 429.** A 429 naming insufficient, usage limit or credit
  balance is SeatUnavailable (marked down), not a rate limit to wait out.
- **Voice ids per provider.** Brand yaml: `voice.voice_id` is the default,
  `voice.voice_ids` maps a tts provider name to its own id. The default brand
  uses Brian (`nPczCjzI2devNBz1zQrb`) on ElevenLabs and George
  (`JBFqnCBsd6RMkjVDRZzb`, RadioPlay's narrator, verified on Elevate and
  Algrow) on `elevate_tts` and `algrow_tts`.
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

Since run 2: a shot shares only the budgets of sources that serve its media
(still shots hold no YouTube share); a search the host throttled before
sending is refunded, so "used" counts sent searches; after the rounds, shots
still short send again the searches that were deferred or throttled, on
sources with budget left after one search per unmatched still shot is held
back for its video frame fallback. Results below the relevance threshold
(one in three of the query's specific words) are kept as fillers after the
matches instead of being dropped; a shot with no match fetches at most
max(1, cap // 4) of them, flagged `off_topic`, and is sourced again on the
next run. Every result a fetch refuses (for
example a YouTube video under the 720p floor) is logged as
`source result dropped` with the shot and the reason.

### 10f. Live checks for the laptop

Run these after filling `.env`. Each sends one or two small requests.

```powershell
python -m vidbush doctor --probe
python -m vidbush.live_probe seats
python -m vidbush.live_probe anthropic
python -m vidbush.live_probe stream wellflow_cheap
python -m vidbush.live_probe stream closerouter
python -m vidbush.live_probe stream wellflow
python -m vidbush.live_probe stream openlux
python -m vidbush.live_probe stream closerouter --role editor
python -m vidbush.live_probe algrow --credits
python -m vidbush.live_probe algrow
python -m vidbush.live_probe elevate --credits
python -m vidbush.live_probe elevate
```

What to look for: `seats` lists `wellflow_cheap:claude-opus-5-5` first for
the writer, CloseRouter's resolved Sonnet 5.5 id first for director and
researcher (this is how its exact id gets verified), `closerouter:x-ai/grok-4.7`
for the editor, and the ids openlux and Wellflow resolved for grok and
gemini (a seat shown "not configured" means its relay lists no such id);
`anthropic` prints `{"ok": true}` twice with a cost above 0 (it fails while
`ANTHROPIC_KEY` is over its usage limit); each `stream` prints a value in a
few seconds; `algrow` and `elevate` write `data/live_probe/<seat>_tts.mp3`, say
`words_returned: false`, and show `aligned_by: whisper` with word times that
look right when you play the file.

## 11. Art layer: illustrated maps, illustrations, clips (run 3)

Designed visuals that are openly illustrations or animations, counted as animation and never as photographs.
Code: `vidbush/art/` and `vidbush/providers/video_elevate.py`; architecture in `docs/ARCHITECTURE.md` (Art layer);
every rule with its reason in `docs/DECISIONS.md` (Run 3: art layer); Elevate's catalogue, credits and limits in
`docs/PROVIDER_NOTES.md` (Elevate catalogue, image and video).

### 11a. What it does

- **Illustrated maps.** The exact map view is rendered with no text and sent to `nano_banana_pro` as a reference
  image. The prompt is written by the director seat from the narration at that moment and the map's places,
  with names and digits removed. The prompt is phrased as a repaint, so the model keeps the outlines. Of three
  variants (two nano_banana_pro, one gemini_flash_image), only those that pass three checks count: registration
  against the true coast, OCR (no text), and a vision call (no painted text or lines). The best of those by
  appeal and fit becomes the painted base. The renderer draws markers, routes, labels, title and credit on top,
  plus the true borders and river names. With no passing variant the plain map stays. Maps that make a precise
  point (route, arrow, shaded region, distance or border words) always stay plain.
- **Illustrations.** For moments no footage can show, in the brand's one style line
  (`illustration_style` in `config/brands/default.yaml`). Each must have no text and no painted bars, and must
  pass the normal judge against its shot window. They replace only AI photo-style stills or weak picks, never
  good real material.
- **Clips.** veo_3_1 image to video from an illustration, a painted map or a photo-style AI still, never from a
  real photograph. The shot holds the still, then moves for 3 to 5 s. A clip is used only when:
  - its first frame matches the still;
  - fixed yes or no vision questions pass (same subject, same layout, no new objects, no text, no morphing);
  - for a map, its middle and last frames register onto the view without moving.

  Otherwise the still keeps its Ken Burns move.

### 11b. Turning it on and off

`config/style_profiles/documentary_default.yaml` now carries `art: {illustrated_maps: 0.5, illustrations: 0.05,
clip_every_s: 90, clip_opening_s: 60}`. Every other profile has no art, because every `ArtProfile` default is
off. To turn it off for a run, set the shares to 0 or remove the `art:` line. Other keys: `map_variants`,
`illustration_variants`, `map_repeat_s` (180), `map_model`, `map_fallback_model`, `illustration_model`,
`clip_min_s`, `clip_max_s`, `clip_from_ai_stills` (false: photo-style AI stills stay still), and `hour_credit_cap` (skip new art while the shared
key has used this many image credits this hour; 0 means no cap).

The video seat is `elevate_video` (veo_3_1, 1080p, 2 credits a clip) first in the `video` role of
`config/seats.yaml`. Set `options.resolution: "720p"` there to halve the credits. `python -m vidbush doctor --probe`
now checks it.

### 11c. Cost and time on Adam's Elevate key

- Map view: three paintings, about 13 credits (5 + 5 + 3) and 2 to 3 minutes.
- Illustration: two paintings, 10 credits.
- Clip: 2 credits at 1080p, about 6 minutes each. The key allows one clip at a time and 40 clips (80 credits)
  an hour.
- At the default shares, a 11 minute documentary with 9 maps, 130 shots and 7 clips spends about 60 + 70 image
  credits and 14 video credits. Clip generation adds about 45 minutes to the visuals stage, because clips run
  one after another.

### 11d. Proving it on a finished project

```powershell
python -m vidbush.art.prove --source "E:\coding projects\Vidbush\output\alaska_run2" --review "E:\coding projects\Vidbush\review_run3" --maps 8 --illustrations 10 --clips 16 --ai-clips 4 --image-cap 300
python -m vidbush.art.sheet --review "E:\coding projects\Vidbush\review_run3"
```

The source project is only read. Everything is written under `<review>\art_project`. The sheet step writes
`art_contact_sheet.png`, `clip_NN.mp4`, `clips.md` and `yield.json` beside it. The director's moments and
picks are cached, so a restart reuses every paid file.

### 11e. Proving run on the Alaska script (2026-11-15)

Run 2's narration and shot list were used read only. Everything is in `E:\coding projects\Vidbush\review_run3\`:
- `art_contact_sheet.png`: 12 numbered items, each with its timestamp and narration line;
- `clip_01.mp4` to `clip_06.mp4`, listed in `clips.md`;
- `yield.json`;
- `art_project\work\art\`: every painting, illustration, clip, check frame and report.

Yield. Every paid file and every check is counted once. A clip counts only when every check it had passed.

| | made | passed every check | on the sheet or numbered |
|---|---|---|---|
| Illustrated maps | 66 paintings of 26 views | 13 paintings; 5 narration moments have a painted map (Alaska twice at two zooms, Bethel, Texas, Cleveland and Los Angeles, Jacksonville) | 5 maps |
| Illustrations | 47 images for 26 shots | 9 shots after the one-scene and lettered-artifact rules (2 collages and 1 classroom wall map dropped) | 7 illustrations |
| Clips | 23 checked (10 illustration, 8 AI still, 5 painted map) | 6: 4 from illustrations, 2 from photo-style AI stills; 0 of 5 map clips | 6 clips |

What the run showed:
- **Wide views work, regional views mostly do not.**
  - Views with a clear coastline at state or country scale work: Alaska at zoom 4.4 and 4.6, Texas, the lower 48, Florida.
  - Regional views at zoom 5 or more, inland views and views with frozen sea mostly fail. The models paint an oblique landscape, a close-up village or sea ice instead of the map, and registration rightly rejects them.
  - The prompt went through five versions (`paint.PROMPT_VERSION` p5): a repaint lead, top-down only, touches as tiny map symbols, and the trace rule repeated at the end.
- **Veo moves the camera on every map clip** (3 to 5 percent zoom or drift) despite the prompt. The near-identity check rejects them, so painted maps keep their push-in over the static painting.
- **Clip vision checks are not deterministic.** The same clip passed one run and failed the next. The sheet counts a clip only when every check passed, and a checked clip is never queued again.
- **My own judgement** of what I would put in a video:
  - the 5 maps;
  - 6 of the 7 illustrations. The biplane hangar at 4:30 shows an aircraft for a line about expired antitoxin.
  - 4 of the 6 clips. Not the two from photo-style AI stills; see the decision that turned those off.
  - The school bus clip adds snow falling inside the bus. The check passed it, but Adam should judge it.

Credits and cost:
- Images: about 570 credits, counted from the files Elevate returned. Tasks that failed server side are not in that count; the key's hourly counters, which other lanes share, showed more.
- Video: about 51 credits (25 clips at 1080p plus one 720p probe).
- Image credits per UTC hour on the shared key: 05:00 about 170, 07:00 412, 08:00 127. New image work stopped at 08:30, apart from one capped round after 09:00 of about 80 credits.
- Text and vision calls: well under 1 USD. The cost report was overwritten by the concurrent proving processes; the last one alone recorded 0.09 USD.

Not proven:
- Nothing has run inside the pipeline end to end. `apply_art` is covered by unit tests only, and no video was rendered with art.
- The moving-base map clip has not reached the screen, because no map clip passed.
- Invented lakes next to a true river still pass registration (the Alaska painting has a seaplane lake).
- The gazetteer resolves "Manhattan" to Manhattan, Kansas. Pipeline maps go through `verify_map`; the proving run skipped that place.
- The illustration scene check and the split-brief filter came late. The earlier rounds were filtered after the fact, not before paying.

See also: `docs/ARCHITECTURE.md` (stage graph, contracts, cache rule),
`docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/BLOCKERS.md`, `docs/SPEC.md`.

## 11. Run 3: AI stills and QA you can trust

### 11a. Reading qa_report.md

The report now opens with the tile audit table: the audit's six numbers
against the Vidrush benchmark and the targets, with a Met column (read from
`audit/audit_summary.json`). Without an audit it opens with one line saying
how to run it (`python -m vidbush audit <id>`, section 12). The pre-render
judge score (`pct_above_threshold`) is no longer a headline: it stays in the
details and in qa_report.json as bookkeeping, because it never tracked the
frames. Then come five lines:

1. Screen time: real (footage plus stills) against generated (AI stills and video), with each class in percent
   and seconds. It follows the picture each shot really uses, so a footage shot that fell back to an AI still
   counts as AI.
2. Rendered shots that fail their frame check, with the most common reasons.
3. The worst five moments, as final.mp4 timestamps with shot id and reason. Look at these first.
4. On-screen text problems over every text item (overlays, lower thirds, boxes, arrows, number overlays, map
   and chart titles and labels, full screen cards): repeats of the narration, early, late, too short to read.
5. The share of rendered shots that pass their frame check ("not measured" when no frames were checked).

How a shot is checked: frames from its own middle (plus its quarters on shots of 4 s or more, plus one where
its overlay text shows) go to the vision seat with the narration spoken during the shot, the lines just
before and after, what kind of picture it is and every text drawn over it. The model answers named checks
(matches the words, text true, clean, one image, no fake face, quality); code adds checks that need the whole
timeline (text off its words, text under 2.5 s, the same label or picture within a minute, a country or
decade subtitle). Any failed check fails the shot; a score fails it only below 40. The "Flagged shots" table
lists the failed checks per shot.

Cost: about 0.045 USD per 11 minute video (Gemini 3.8 Flash, the middle frame at high detail).

### 11b. Calibrating QA against a frame review

```powershell
.venv\Scripts\python.exe scripts\qa_calibration.py --project "E:\coding projects\Vidbush\output\alaska_run2" `
  --review "E:\coding projects\Vidbush\review_run2\review_run2.json" `
  --extras "E:\coding projects\Vidbush\review_run2\review_run2_extras.json" `
  --settings-root C:\projects\vidbush_tmp --out "E:\coding projects\Vidbush\scratch\run3_qa_cal" --tag v4
```

It reads the project only, writes frames, verdicts, costs and logs to `--out`, and prints recall (share of
review defects QA fails) and false alarms (share of review passes QA fails); `calibration_<tag>.json` lists
every miss and false alarm with QA's reason. `--reuse-frames` skips frame extraction, `--reuse-verdicts`
re-scores saved verdicts without model calls. Run 2 results are in DECISIONS.md under "Run 3: stills and QA".
A review in another shape needs its own `review_labels`.

### 11c. AI stills

- The prompt comes from a scene brief of the narration spoken during the shot: who or what acts, the action,
  the object, the place with its look, the period, one moment. Stub intents never become prompts. Brief and
  prompt are stored on the candidate (`brief`, `prompt` in shotlist.json).
- Every AI still gets one extra vision call with seven fixed questions (`judge/ai_checks.py`); a "no" fails
  the take and its reason steers the next take (up to `judge.ai_retries`, default 2). The answers are in the
  candidate's `checks`.
- A shot whose narration is about a named real person gets a context still and the flag
  `real_person_photo_wanted`: give it a real photo through the review page (replace with your file) when you
  have one.
- Each AI still now costs one more editor call and one more vision call; on the Elevate subscription the image
  itself stays free. Expect about 3 to 4 minutes per take of wall clock (image about 100 s, judge and check
  about 80 s).

To redo chosen AI shots of a finished project into a scratch copy (the project is not touched):

```powershell
.venv\Scripts\python.exe scripts\regen_ai_shots.py --project "E:\coding projects\Vidbush\output\alaska_run2" `
  --out "E:\coding projects\Vidbush\scratch\run3_regen" --shots shot_068,shot_087 --settings-root C:\projects\vidbush_tmp
```

## 12. V2 measurement and replay (2026-11-17)

The tile audit is now the quality measure. It judges the finished frame
against the words heard, never the pipeline's own scores. Stage snapshots,
the reply cache and the cost cap inside stages make reruns cheap and
comparable. Decisions are in `docs/DECISIONS.md` under "V2: measurement".

Run every command from the repo in PowerShell:

```powershell
cd C:\projects\vidbush_tmp
```

Results of night 1 (audit with the shipped prompt; review = the hand review in docs/FABLE_REVIEW.md):

| Video | real (audit/review) | AI | designed | on_words | noticeable | overlay noticeable |
|---|---|---|---|---|---|---|
| run 2 | 22.7 / 26.9 | 71.0 / 70.1 | 6.0 / 2.7 | 43.8 / 35.0 | 39.3 / 49.2 | 53.1 / 64.6 |
| run 3 | 67.1 / 71.9 | 26.0 / 25.7 | 6.9 / 0.9 | 58.6 / 52.3 | 22.1 / 23.6 | 43.3 / 54.7 |
| Vidrush | 56.4 / 54.7 | 2.6 / 2.0 | 40.7 / 42.4 | 71.3 / 64.2 | 18.6 / 18.9 | 24.8 / 21.1 |
| dunkirk_base | 65.9 / none | 26.8 | 6.8 | 55.0 | 27.3 | 28.1 |

- Proof bar (within 5 points of the review): real share passes on all three. noticeable
  passes on run 3 and Vidrush and fails on run 2 (-9.9). on_words fails on all three
  (+6.3 to +8.8: the audit is more generous than the review about a picture on the words).
  Overlay noticeable runs about 11 points low on the Alaska runs. A stricter wording put
  on_words 6 to 10 points low; a halfway wording was no better and was reverted.
- The calibration page (40 run 3 tiles) is the next step: E:\coding projects\Vidbush\audit\calibration_run3.html.
  Adam marks it, then `python -m vidbush audit-calibrate <marks.json>`; the audit is trusted at 34 of 40.
- dunkirk_base (seed 1940, cost cap 5): done, 2.04 USD, wall clock about 3 h 10 min (judge
  stage 7117 s, sourcing 2408 s). Audit 0.22 USD, 245 s. Worst problems it found: wrong
  period stand-ins (WWI trenches for RAF lines, Napoleonic reenactors for 1940 Lille),
  narrow portrait photos with blurred side bars, AI faces for named people (Goring), a
  stock watermark at 0:06-0:14, and an empty label box near 6:20. Both image providers'
  content filters blocked some war stills; those shots fell back without an AI picture.
- Spend tonight about 5.6 USD of the 8 USD cap. Full suite after the merge: 1435 passed,
  17 skipped (baseline 1383 passed, 17 skipped).

### 12a. Tile audit

One frame every 2 s of final.mp4 (tile t, frame at t+1). The `auditor` seat
(Sonnet 5.5) sees the frame, the words spoken in [t-1, t+2] and 10 s of
narration before and after. For a project, a tile whose shot uses a
generated picture is told so, and an AI kind replaces a real or other guess
(the guess stays as `kind_seen`). Rows are appended as they finish, so a
crash loses nothing, and a rerun pays only for tiles whose prompt, model,
words or video changed.

Audit a project (reads `final.mp4`, `work/narration.json` and
`work/shotlist.json`):

```powershell
.venv\Scripts\python.exe -m vidbush audit alaska_run3
```

Try it on a slice first (40 tiles spread over the video):

```powershell
.venv\Scripts\python.exe -m vidbush audit alaska_run3 --limit 40
```

Audit a bare video against a transcript (`startMs`/`endMs` words) or a
`narration.json`:

```powershell
.venv\Scripts\python.exe -m vidbush audit C:\projects\CaseFile\inbox\alaska\vidrush-alaska.mp4 `
  --script C:\projects\CaseFile\docs\evidence\vidrush-grammar\transcript.json
```

Other flags: `--out DIR`, `--every 2`, `--workers 8`. A run with no valid
tile writes no summary and exits 1. Cost so far is in
`audit_cost_report.json`, never in the project's cost report.

Compare an audit with reference tile rows (joined on t):

```powershell
.venv\Scripts\python.exe -m vidbush audit-compare "E:\coding projects\Vidbush\output\alaska_run3\audit\audit.jsonl" `
  "E:\coding projects\Vidbush\fable_review\rows\run3_01_06.jsonl" `
  "E:\coding projects\Vidbush\fable_review\rows\run3_07_12.jsonl" `
  "E:\coding projects\Vidbush\fable_review\rows\run3_13_17.jsonl"
```

It prints the six numbers of both sets, the difference, and per-tile
agreement (match, on_words, noticeable, kind group). `--json` prints the
raw result.

### 12b. Calibrating the audit against your own eyes

Build the 40 tile page (20 flagged, 20 not, spread over the video). The page
never shows the audit's verdict; the verdict goes to a separate key file:

```powershell
.venv\Scripts\python.exe -m vidbush audit-sheet "E:\coding projects\Vidbush\output\alaska_run3\audit\audit.jsonl" --name run3
```

Open `E:\coding projects\Vidbush\audit\calibration_run3.html`, mark every
tile, press Export. It downloads `calibration_marks.json`. Then:

```powershell
.venv\Scripts\python.exe -m vidbush audit-calibrate $HOME\Downloads\calibration_marks.json
```

It finds the key for the same video by itself (or pass `--key`). Tile ids
carry a video hash, so marks for another render are refused. TRUSTED needs
34 of 40 agreements with every tile marked; the exit code is 1 otherwise.
Each disagreement is listed with its time, your mark, the audit's defect and
your comment.

### 12c. Plan replay (no render)

Measure a stored shot plan against its narration, without rendering or any
model call:

```powershell
.venv\Scripts\python.exe -m vidbush plan-replay alaska_run3
```

It reads `work/shotlist.shotlist.json` (the director's plan) when it exists,
else `work/shotlist.json`; `--shotlist PATH` picks another one and
`--json OUT` writes the report. It prints: picture shots whose anchor words
fall inside their own window (and how many are early, late or have no
anchor), cuts per minute, median shot length, screen time by kind (real
footage, real still, AI, designed or motion graphic), and the beats (or
clauses, when `work/clauses.json` exists) with no picture shot. Nothing is
written into the project folder.

### 12d. Runs, cost cap and replays

Start a new project with a cost cap of 5 USD:

```powershell
.venv\Scripts\python.exe -m vidbush run "E:\coding projects\Vidbush\output\alaska_run3\inputs\05-alaska-size-narration.md" --id alaska_v2 --cost-cap 5
```

`run` refuses an id that already exists. To delete it and start again:

```powershell
.venv\Scripts\python.exe -m vidbush run "E:\coding projects\Vidbush\output\alaska_run3\inputs\05-alaska-size-narration.md" --id alaska_v2 --fresh --cost-cap 5
```

The cap is checked before every live paid call, so it holds inside a stage.
A stage stopped inside stays pending. The project ends `needs_review` with
the reason in `status_note` and `work/cost_cap.json`. Finish it with a
higher cap (stored replies cost nothing):

```powershell
.venv\Scripts\python.exe -m vidbush resume alaska_v2 --cost-cap 10
```

Re-score a finished project from its sourcing snapshot. Unchanged judge
prompts are served free from the reply cache, so only changed prompts are
paid for; every stage after judge reruns too:

```powershell
.venv\Scripts\python.exe -m vidbush resume alaska_v2 --force judge
```

`--force sourcing` re-sources from the director's plan the same way. Each
shot list stage writes `work/<stage>.shotlist.json`, and a forced stage
starts from the snapshot of the stage before it. If you edited the shot list
since the last stage (review fixes, an edit op), an unforced stage keeps your
edit. A forced stage keeps a copy as `work/operator_edit_<ts>.shotlist.json`
before it starts from its snapshot.

Ask every model live (stored replies are neither read nor written). The
flag goes before the subcommand:

```powershell
.venv\Scripts\python.exe -m vidbush --no-reply-cache resume alaska_v2 --force judge
```

`$env:VIDBUSH_REPLY_CACHE = "0"` or `reply_cache: false` in
`config/vidbush.yaml` does the same. The cache key ignores `max_tokens`:
clear the reply cache folder after changing an output cap.

Every run is recorded in `project.json` under `runs` (commit, branch, dirty
flag, argv, forced stages, cost cap, seat models, config hashes). Every
stage keeps a `history` of its attempts with times, status and commit.

### 12e. Where outputs go

- Project audit: `<project>\audit\` (for example
  `E:\coding projects\Vidbush\output\alaska_run3\audit\`): `audit.jsonl`,
  `audit.md`, `audit_summary.json`, `audit_cost_report.json`, `tiles\`.
- Bare video audit: `<parent of the output folder>\audit\<video stem>\`
  (here `E:\coding projects\Vidbush\audit\vidrush-alaska\`).
- Calibration page and key: `E:\coding projects\Vidbush\audit\calibration_<name>.html`
  and `calibration_<name>_key.json`.
- Shot list snapshots: `<project>\work\<stage>.shotlist.json` and
  `work\snapshots.json`.
- Reply cache: `reply_cache\` beside the source cache folder (the parent of
  `SOURCE_CACHE_FOLDER`), one file per request hash under
  `<role>\<key[:2]>\`.

### 12f. Reading audit.md

The first line gives the tile count, how many tiles carry an overlay, the
model and the audit cost. Tiles with no usable verdict are left out and
counted. Then a table of six numbers, each a share of tiles:

| Measure | What it counts | Vidrush benchmark | Target |
|---|---|---|---|
| Real share | real footage, real stills, documents | 55% | none |
| AI share | AI images and AI-looking video | 2% | 5% or less |
| Designed share | maps, charts, text cards, designed cards | 42% | none |
| Picture on_words | the picture shows what the words are about | 64% | 64% or more |
| Noticeable | a viewer would notice a problem at normal speed | 19% | 19% or less |
| Overlay tiles noticeable | noticeable, among tiles with editor text or graphics | 21% | under 15% |

The benchmark is the Vidrush render of the Alaska script (FABLE_REVIEW
section 3). An empty share reads n/a, never 0. Below the table are the ten
worst stretches: runs of consecutive noticeable tiles, longest first, with
their final.mp4 time range and the defects named. Look at those first.

The same numbers head `qa_report.md` (section 11a) once the audit exists.
Acceptance (`python -m vidbush check <id>`) fails a project whose audit
shows more than 40 percent noticeable; a missing audit is reported, not
failed.

### 12g. Other changes tonight

- An unusable judge reply is asked again twice with a fresh reply, then
  left unjudged (score None), never scored 50. An unjudged candidate ships
  only when nothing on the shot is judged, flagged `unjudged`.
- A requery model error falls back to the heuristic queries; an outage
  ("all providers failed") is raised and writes no queries.
- Image credits are recorded from per-job charge keys only. A provider that
  charges over 5 times another per unit for one model in the same run is a
  warning in the cost report.
- Source replay reads the search cache under the exact query the run sent
  and ignores the cache TTL.
## 13. V2 sources: finding the specific real picture (branch v2/sources, 2026-11-17)

Lane D of the Fable review (docs/FABLE_REVIEW.md on fable-review). Everything is in `vidbush/source/`;
models and config changes are additive. Decisions: DECISIONS.md "V2: sources"; quirks: PROVIDER_NOTES.md "V2
sources".

### 13a. New sources

| Source | Module | Key | Ran live here | Licence |
|---|---|---|---|---|
| Web image search | `webimage.py` | SERPER_API_KEY, else FIRECRAWL_API_KEY, else the search seat (serper, brave, tavily) | yes, Firecrawl | always "unknown (web image search)", flagged licence_unknown |
| Openverse | `openverse.py` | none (anonymous) | yes | CC and public domain as given |
| Chronicling America | `chronam.py` | none | search: no (www.loc.gov challenges this laptop); page images and word crops from tile.loc.gov: yes | public domain |
| Wikimedia by entity | `wikimedia.py` | none | yes | as on Commons |
| Library of Congress | `loc.py` (changed) | none | no: every request is a Cloudflare challenge; pause cut to 600 s so it never stalls a stage | as before |

Missing or broken keys on this laptop: SERPER_API_KEY is not set; SEARCH_API_KEY (tavily) answers 401, so the
research seat's web search fails too. Add a Serper key to `.env.local` to use Serper (it is tried first).

### 13b. How queries reach sources

- A shot may carry `source_queries` ({class: [queries]}) and `real_thing` ({name, kind, date, place}).
  `registry.source_shot(shot, ctx, queries=...)` also takes `{"real_thing": {...}, "queries": {...}}`.
  Classes: archive, encyclopedic, newspaper, web_image, web_video, stock (aliases: web, video).
- Each adapter serves some classes (`adapter.classes`): YouTube web_video; Wikimedia encyclopedic and archive;
  Internet Archive archive and web_video; NARA, NASA, LoC archive; Chronicling America newspaper and archive;
  web image search web_image and newspaper; Openverse web_image, encyclopedic, stock, archive; Pexels and
  Pixabay stock; the local library all. A flat `queries` list still goes to every source.
- `registry.source_pool(shots, ctx)` sources a stretch of shots into one pool (every asset fetched once) for an
  edit step that assigns from a pool. Nothing calls it yet.
- Every candidate now carries `query`, `rank`, `description`, `date`, `page_url`, `found_by` (also in the
  sidecar). The judge prompt and the planner's edit catalogue do not read them yet (other lanes' files).

### 13c. Proof (Part C)

Harness (scratch, not in the repo): `E:\coding projects\Vidbush\scratch\v2src\partc_run.py` with the things in
`partc_things.py`; sheets by `partc_sheet.py`. Output in `E:\coding projects\Vidbush\sources_v2\`
(`results_<script>.json`, `review\<script>\<id>_<media>.jpg` with every candidate, `contact_sheet_alaska.jpg`,
`contact_sheet_dunkirk.jpg`). Each thing is a still shot sourced live with a real_thing structure written from
the narration; four were also sourced as footage.

Alaska (the review's 12): the real thing came back for 9 in pass 2 and 8 in the final pass (A04, the Billboard
jukebox chart, was web rank 3 and fell outside the 8 fetched assets once Commons returned more for the song).
Not found: the Anchorage Daily Times apology editorial, the "Texas No Longer Largest" headline, the Texas
Senate song amendment (a real 1959 Senate Journal page came back, the wrong page). Dunkirk (8 chosen): 8 of 8.

### 13d. Known limits

- `replay.py` (lane A's file) does not copy `classes`, `matches_description` or `uses_real_thing` onto its
  OfflineAdapter and does not list webimage, openverse, chronam: a replay routes and screens differently from
  a live run and misses the entity sources' caches. Copy the three attributes with getattr and add the three
  adapters there.
- The judge's prompt line and qa.py SOURCE_LABELS do not know the new sources or the provenance fields.
- An 8-asset cap per shot costs time on footage: one shot fetching 8 YouTube and archive assets took about 20
  minutes cold in the proof (about 2 minutes from the cache). Watch the sourcing stage time on the next run.
- YouTube answered a bot check after four proof passes in about three hours with no cookies set; the adapter
  paused it for 30 minutes as designed. Set YOUTUBE_COOKIES_FROM_BROWSER for long sessions.
- A search that sends a second request (the yearless retry, the entity path) is charged once to the budget;
  a throttle on the second request refunds the whole charge (low impact).
- Web image results are often stock previews with watermarks (alamy, getty) or small (Facebook, Reddit). The
  licence is unknown for all of them; the near-duplicate filter keeps the openly licensed copy when the same
  photo also arrives from Commons or Openverse.
## 14. Designed layouts and panels (V2, branch v2/layouts)

### 14a. What it does

A shot can carry a designed layout (`Shot.layout`, written by the planner). Eleven kinds, all drawn by our own
Python renderer in one house style (`vidbush/mg/house.py`): split comparison, number over footage, name card,
list slideshow, date title, route map, equals card, chart, table, timeline and document focus. Every element is
timed to the narration words it belongs to and never appears more than 0.3 s before them. Route maps, charts,
tables and timelines can also be shown as a panel over the shot's own picture, in four ways: full, panel,
full_then_dock (opens at 75 percent, shrinks into a corner) and panel_then_expand (grows to full, its stage 2
elements wait for the growth). The panel's place is read from the picture beneath (faces, text, the subject),
varies from the previous panel, and its reason is written to the shot's `layout_notes`. Old map and chart
insets in existing plans are converted to placed, word-timed panels automatically.

Also new: bare full-frame text cards and centre callouts are gone, label boxes never show empty, stills move
toward their subject (or hold when short), a still narrower than half the frame is framed on the house ground
instead of blurred bars, and a matched look can bring sources toward one exposure and colour.

### 14b. Switches (style profile)

- `letterbox.enabled`, `captions.enabled`: the format switches (defaults unchanged: letterbox and captions on).
  Every layout fits both ways.
- `colour.match_look` (off), `colour.match_strength` (0.6), `colour.brand_grade` ("", or `house_warm`,
  `house_cool`).
- `Brand.house_style`: overrides of the house colours and motion, e.g. `{accent: "#E8A33D"}`.

### 14c. Render cost (measured on the laptop, render seconds per clip second)

Full layouts: equals card 0.5, date title 1.0, name card 0.8, number over footage 0.8, split comparison 1.3,
list slideshow 1.4, timeline 0.4 to 0.8, chart 0.4, table 0.5, route map 1.1 to 2.3, document focus 2.5 to 5
(OCR on first use, cached after). Panels: 0.6 to 1.5 per clip second plus placement analysis (4 frames, about
1 s per panel). No model call is made by any layout.

### 14d. Proving it without a pipeline run

Scripts in `E:\coding projects\Vidbush\layouts\scripts`: `prove.py alaska` (hand-written specs over run 3's
narration and candidates, `specs_alaska.py`), `prove.py other` (invented war, science and business moments,
`specs_other.py`, pictures from Wikimedia Commons in `layouts\other_pictures`), `placement_proof.py` (one panel
over five pictures), `check.py <tag>` (a frame every 0.5 s with each element's word time), `make_reel.py`
(`layouts\layout_reel.mp4` and `layouts\layout_sheet.png`). Each runs the real motion stage
(`layouts.stage.render_shot_layouts`) and the real segment renderer.

### 14d2. Documents

`document_focus` takes `purpose`: evidence (recognise the page, move to the passage, mark it as read, source label),
period (atmosphere: a panel or slow push, no marks) or subject (full frame, each named part marked in spoken
order), and the four presentations. "mark" elements name the words on the page to mark and the narration words
they are marked on. Proofs: `layouts\scripts\specs_docs.py all`, then `check.py docs` and `check.py docs_other`.
OCR of a large scan takes 20 to 60 s on first use (cached per file after).

### 14e. What the planner must supply

`LayoutSpec` with elements in spoken order and anchors that are the exact narration words (numbers as spoken:
"twenty-nine people"). Role names per kind: `models.LAYOUT_ROLES`. Start a full layout near its first anchored
word and hold it 3 to 7 s (the notes flag anything else). A name card needs a real portrait candidate; a
document focus needs a real document whose page carries the quote.

## 15. V2 integration (branch v2/base, night 2)

v2/base = v2/audit + v2/sources + v2/layouts + v2/planner. Decisions: DECISIONS.md "V2: integration".

### 15a. The layout contract

The planner (`vidbush/plan/emit.py`) emits the layouts lane's `models.LayoutSpec`: roles from `LAYOUT_ROLES`,
pictures as candidate ids on elements, a found document as a `document` element (plus `source`), background
elements for number_over_footage and date_title from the clause's own pick, timeline `id` from the planner's
timeline_id, `purpose` only on document_focus. A full layout's cut is moved to its first anchored word
(`start_full_layouts_on_words`); panels get `layout_at` from the renderer. Tests: `tests/test_plan_contract.py`.

### 15b. YouTube's paid second route

`vidbush/source/youtube_api.py` (provider notes: PROVIDER_NOTES.md "video-download-api.com"). Free yt-dlp first;
the first bot check sets `youtube.ROUTE.name = "api"` for the rest of the process and the same video is fetched
again on the paid route. Without a key the old 30 minute pause stays. Config: `sources.youtube_api` (on),
`sources.youtube_api_budget_usd` (0.50 per run). Downloads per route this process: `youtube.ROUTE.counts`;
paid jobs: `youtube_api.LEDGER` (spent, jobs), written into cost_report.json at each sourcing usage note.
Live proof: `E:\coding projects\Vidbush\scratch\vda_live_test.py` (two videos, 1280x720, licence kept).

## 16. V2 picker: describe once, pick by comparison (branch v2/picker, night 2)

Decisions: DECISIONS.md "V2: picker". Architecture: ARCHITECTURE.md "Clause-first planner", step 3.

### 16a. What runs

- `vidbush/judge/describe.py`: one describer call per asset (frames across the clip, or the still), cached
  by content hash plus clip range in `DESCRIBE_CACHE_FOLDER` (tonight `E:\coding projects\Vidbush\describe_cache`,
  572 records). Stills also get rapidocr words. Prefilter rejects and failed AI checks never cost a call.
- `vidbush/plan/edit.py`: the picker seat, one call per scene, at most 40 assets (one 640 px image each),
  start time per pick; code checks listed in DECISIONS.
- `vidbush/judge/ladder.py`: re-source, restate as layout, hold, AI still; `filled_by` per clause.
- `vidbush/judge/pick_stage.py`: the live judge stage for a clause-planned project (profile `planner: clause`).
  Not yet run end to end in a live project.
- Seats: `describer` and `picker` in config/seats.yaml. openlux Sonnet entries carry
  `extra_body: {reasoning_effort: low}`; without it one scene's pick cost 193k tokens.

### 16b. Replay commands (no render)

```
set PYTHONPATH=.  (worktree root)  SOURCE_CACHE_FOLDER=E:\coding projects\Vidbush\vidbush_tmp\data\source_cache
DESCRIBE_CACHE_FOLDER=E:\coding projects\Vidbush\describe_cache
python -m vidbush.judge.pick_replay "E:\coding projects\Vidbush\output\alaska_run3" "E:\coding projects\Vidbush\picker\alaska" ^
  --env C:\projects\vidbush_tmp --direct-from "E:\coding projects\Vidbush\plan_v2\run" ^
  --extra "E:\coding projects\Vidbush\sources_v2\results_alaska.json" "E:\coding projects\Vidbush\sources_v2\alaska_input.md" ^
  --sheets s01,s08 --previous "E:\coding projects\Vidbush\plan_v2\run" --budget 3
```

Dunkirk: project `output\dunkirk_base`, `--scenes s01,s03 --direct-from picker\dunkirk_before` (director plans
made tonight on v2/base), `--extra sources_v2\results_dunkirk.json sources_v2\dunkirk_input.md`,
`--previous output\dunkirk_base` (the old judge path's shot list). `--salt X --no-ladder` repeats the picker
call with a new cache key. `--resource` turns on ladder step 1 (live searches and downloads).

A replay is free only when every prompt is unchanged: a changed restate in an early scene changes which
assets later scenes see as used, and their picker calls are asked again.

### 16c. Results (stored data, no render)

Outputs: `E:\coding projects\Vidbush\picker\alaska`, `picker\dunkirk`, `picker\alaska_s01_repeat`.
Independent judge (one AI agent, one still per clause at mid-speech, both columns judged together):

| Scene | Earlier: on words / early-late / wrong / related / nothing | Tonight | Wrong document, look-alike (earlier to tonight) |
|---|---|---|---|
| Alaska s01 (23 clauses; earlier = last night's planner edit) | 11 / 0 / 0 / 10 / 2 | 12 / 2 / 0 / 6 / 3 | 1, 1 to 0, 0 |
| Alaska s08 (20) | 8 / 1 / 2 / 7 / 2 | 10 / 0 / 0 / 9 / 1 | 1, 0 to 1, 0 (the same park map in both) |
| Dunkirk s01 (19; earlier = old judge render) | 3 / 0 / 2 / 14 / 0 | 9 / 0 / 2 / 3 / 5 | 0, 2 to 0, 0 |
| Dunkirk s03 (23) | 11 / 3 / 0 / 9 / 0 | 8 / 2 / 0 / 7 / 6 | 0, 1 to 0, 2 (real Mole, photographed in 2018) |

"Nothing" tonight includes planned AI stills, which a replay does not generate (Alaska s01 3, s08 1,
Dunkirk s03 4), and in Dunkirk s01 five clauses the ladder could not fill (a real event: no AI allowed).

Alaska whole script (11 scenes, 200 clauses): 180 shots, 16.2 cuts per minute, median 3.36 s. Of 156 clauses
that need a picture: picked 63, restated 77 (text over footage 55, number 10, date 6, route map 6), AI still 7
(22.0 s, longest run 2), hold 1, nothing 8. 38 assets refused by their records (marks, subtitles, logos);
461 of 901 pool assets left out by the 40-asset cap. Documents: 5 planned, no page in the pool shows the
quoted words, none accepted (the 1958 front pages that last night's editor used were refused). Labels: 2,
both from the record (Toksook Bay, Utqiagvik). Last night (same director plans): 6 AI stills, 15.0 s, run 2.

Time and cost, whole Alaska script, cold: about 440 describe calls, 19 pick calls, 20 restate calls, about
33 minutes wall clock with 8 describe workers, about 2.8 USD on openlux Sonnet (closerouter was down all
night). The old judge stage: about 170 minutes. From cache: describe 0.3 s per scene, a cached pick about 3 s.

Run to run (Alaska s01 picked twice, new cache key): of 16 shots, 13 identical decisions, 3 changed between a
pick and nothing fits, none changed to a different asset.

### 16d. Read before trusting

- The measured runs used the code before the review fixes (6ab2fef) and before an unfound quoted document went
  down the ladder; those fixes are unit-tested, not replayed.
- Restating is the most used ladder step and often types the clause over loosely related stock; the judge
  marked those related, and twice the typed line belonged to the next clause (now refused in code).
- Ladder step 1 (re-source) and the live judge stage path were not run on real data tonight.
- Spend: 5.94 USD booked across all runs and probes, over the 5 USD limit. 1.17 USD of that is wellflow's
  booked estimate (0.01 USD per high image, the provider returns no cost) for 52 describe calls during the
  closerouter outage; provider-reported spend is 4.77 USD.
- Newspaper pages: Chronicling America was not reached through Firecrawl (loc.gov asks clients to pause on a
  CAPTCHA). The web image route phrased for newspaper pages is tested offline only.

### 16e. Adam's review

Contact sheets, earlier (left) against tonight (right), one row per clause:
`E:\coding projects\Vidbush\picker\alaska\review\review_s01_1.jpg`, `review_s01_2.jpg`,
`review_s08_1.jpg`, `review_s08_2.jpg`.
Read down the picks for the two Alaska scenes and say whether each picture now shows what its words say.
